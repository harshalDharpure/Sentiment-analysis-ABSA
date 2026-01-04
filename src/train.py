"""
Training loop for Dimensional ABSA 2026 Track A.

Key features:
    - Prompt-based dataset using `DimABSADataset`
    - Dual-head regression with MSE loss
    - Pearson correlation metrics for Valence and Arousal
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.augmentation import generate_synthetic
from src.config import get_default_config
from src.dataset import DimABSADataset
from src.model import DimABSAModel
from src.utils import (
    get_device,
    num_parameters,
    pearson_corr,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    cfg = get_default_config()
    parser = argparse.ArgumentParser(description="Train DimABSA model (Track A).")
    parser.add_argument("--train-file", type=str, required=True, help="Path to labeled train JSONL.")
    parser.add_argument("--eval-file", type=str, default=None, help="Optional eval/dev JSONL.")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=cfg.training.num_epochs)
    parser.add_argument("--lr", type=float, default=cfg.training.learning_rate)
    parser.add_argument("--batch-size", type=int, default=cfg.training.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=cfg.training.eval_batch_size)
    parser.add_argument("--max-length", type=int, default=cfg.training.max_seq_length)
    parser.add_argument("--warmup-ratio", type=float, default=cfg.training.warmup_ratio)
    parser.add_argument("--weight-decay", type=float, default=cfg.training.weight_decay)
    parser.add_argument("--max-grad-norm", type=float, default=cfg.training.max_grad_norm)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true", help="Apply simple synonym augmentation.")
    parser.add_argument("--augment-ratio", type=float, default=0.3, help="Fraction of train examples to augment.")
    parser.add_argument("--pretrained", type=str, default=cfg.model.pretrained_model_name)
    parser.add_argument("--hidden-dim", type=int, default=cfg.model.hidden_dim)
    parser.add_argument("--dropout", type=float, default=cfg.model.dropout)
    return parser.parse_args()


def maybe_augment_dataset(dataset: DimABSADataset, ratio: float) -> DimABSADataset:
    """
    Create an augmented copy of a subset of the dataset.
    """
    cutoff = int(len(dataset) * ratio)
    for idx in range(cutoff):
        ex = dataset.examples[idx]
        aug_text, aug_aspect, valence, arousal = generate_synthetic(
            ex.text, ex.aspect, ex.valence, ex.arousal
        )
        dataset.examples.append(
            type(ex)(
                example_id=f"{ex.example_id}__aug",
                text=aug_text,
                aspect=aug_aspect,
                valence=valence,
                arousal=arousal,
            )
        )
    return dataset


def create_dataloader(
    file_path: str,
    tokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    include_labels: bool,
) -> DataLoader:
    ds = DimABSADataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_length=max_length,
        include_labels=include_labels,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate(model: DimABSAModel, dataloader: DataLoader, device: torch.device) -> dict:
    model.eval()
    mse = torch.nn.MSELoss()
    val_losses = []
    aro_losses = []
    val_preds, val_targets = [], []
    aro_preds, aro_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            valence_pred = outputs["valence"].squeeze(-1)
            arousal_pred = outputs["arousal"].squeeze(-1)

            valence_true = batch.get("valence")
            arousal_true = batch.get("arousal")
            if valence_true is None or arousal_true is None:
                continue
            valence_true = valence_true.to(device)
            arousal_true = arousal_true.to(device)

            v_loss = mse(valence_pred, valence_true)
            a_loss = mse(arousal_pred, arousal_true)
            val_losses.append(v_loss.item())
            aro_losses.append(a_loss.item())

            val_preds.extend(valence_pred.cpu().tolist())
            val_targets.extend(valence_true.cpu().tolist())
            aro_preds.extend(arousal_pred.cpu().tolist())
            aro_targets.extend(arousal_true.cpu().tolist())

    metrics = {
        "valence_mse": float(torch.tensor(val_losses).mean()) if val_losses else 0.0,
        "arousal_mse": float(torch.tensor(aro_losses).mean()) if aro_losses else 0.0,
        "valence_pearson": pearson_corr(val_preds, val_targets) if val_preds else 0.0,
        "arousal_pearson": pearson_corr(aro_preds, aro_targets) if aro_preds else 0.0,
    }
    model.train()
    return metrics


def train() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    train_loader = create_dataloader(
        file_path=args.train_file,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        include_labels=True,
    )

    if args.augment:
        train_loader.dataset = maybe_augment_dataset(train_loader.dataset, args.augment_ratio)  # type: ignore

    eval_loader: Optional[DataLoader] = None
    if args.eval_file:
        eval_loader = create_dataloader(
            file_path=args.eval_file,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            shuffle=False,
            include_labels=True,
        )

    model = DimABSAModel(
        pretrained_model_name=args.pretrained,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    total_params, trainable_params = num_parameters(model)
    print(f"Total params: {total_params/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if len(train_loader) == 0:
        raise ValueError("Training dataset is empty. Ensure labels (Valence/Arousal) are present.")

    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    mse = torch.nn.MSELoss()
    best_eval_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            valence_true = batch.get("valence")
            arousal_true = batch.get("arousal")
            if valence_true is None or arousal_true is None:
                continue  # skip unlabeled rows
            valence_true = valence_true.to(device)
            arousal_true = arousal_true.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            valence_pred = outputs["valence"].squeeze(-1)
            arousal_pred = outputs["arousal"].squeeze(-1)

            loss = mse(valence_pred, valence_true) + mse(arousal_pred, arousal_true)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            if step % 20 == 0:
                avg_loss = running_loss / step
                print(f"Epoch {epoch} Step {step}/{len(train_loader)} Loss {avg_loss:.4f}")

        # Evaluation
        if eval_loader is not None:
            metrics = evaluate(model, eval_loader, device)
            eval_loss = metrics["valence_mse"] + metrics["arousal_mse"]
            print(
                f"[Eval] Epoch {epoch} "
                f"V-MSE {metrics['valence_mse']:.4f} "
                f"A-MSE {metrics['arousal_mse']:.4f} "
                f"V-Pearson {metrics['valence_pearson']:.4f} "
                f"A-Pearson {metrics['arousal_pearson']:.4f}"
            )
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(model, best_ckpt)
                print(f"Saved new best checkpoint to {best_ckpt}")
        else:
            # Always save last epoch if no eval set
            save_checkpoint(model, best_ckpt)

    print("Training completed.")
    if best_ckpt.exists():
        print(f"Best checkpoint at: {best_ckpt}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()



