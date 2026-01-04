"""
Inference & submission utilities for Dimensional ABSA 2026 Track A.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.dataset import DimABSADataset
from src.model import DimABSAModel
from src.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for DimABSA 2026 Track A.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to JSONL file (dev/test).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--output-jsonl", type=str, default="submission/predictions.jsonl")
    parser.add_argument("--submission-zip", type=str, default="submission.zip")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pretrained", type=str, default="microsoft/deberta-v3-large")
    return parser.parse_args()


def create_dataloader(input_file: str, tokenizer, max_length: int, batch_size: int) -> DataLoader:
    ds = DimABSADataset(
        file_path=input_file,
        tokenizer=tokenizer,
        max_length=max_length,
        include_labels=False,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def run_inference(args: argparse.Namespace) -> List[dict]:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    dataloader = create_dataloader(args.input_file, tokenizer, args.max_length, args.batch_size)

    model = DimABSAModel(pretrained_model_name=args.pretrained)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Collect predictions grouped by ID
    id_to_aspects: dict = {}
    
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
            valence_pred = outputs["valence"].squeeze(-1).cpu().tolist()
            arousal_pred = outputs["arousal"].squeeze(-1).cpu().tolist()

            ids = batch["id"]
            aspects = batch["aspect"]
            for i in range(len(valence_pred)):
                # Extract original ID (before __idx suffix)
                full_id = ids[i]
                original_id = full_id.split("__")[0] if "__" in full_id else full_id
                
                v = float(valence_pred[i])
                a = float(arousal_pred[i])
                # Clamp to [1.0, 9.0] range
                v = max(1.0, min(9.0, v))
                a = max(1.0, min(9.0, a))
                
                if original_id not in id_to_aspects:
                    id_to_aspects[original_id] = []
                
                id_to_aspects[original_id].append({
                    "Aspect": aspects[i],
                    "VA": f"{v:.2f}#{a:.2f}"
                })
    
    # Format as submission: {"ID": "...", "Aspect_VA": [...]}
    results: List[dict] = []
    for ex_id, aspect_list in id_to_aspects.items():
        results.append({
            "ID": ex_id,
            "Aspect_VA": aspect_list
        })
    
    return results


def write_jsonl(records: List[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def package_submission(jsonl_path: str | Path, zip_path: str | Path) -> None:
    jsonl_path = Path(jsonl_path)
    zip_path = Path(zip_path)
    root_dir = jsonl_path.parent
    archive_name = zip_path.with_suffix("").as_posix()
    shutil.make_archive(archive_name, "zip", root_dir=root_dir)
    print(f"Created submission archive at {zip_path}")


def main() -> None:
    args = parse_args()
    results = run_inference(args)
    write_jsonl(results, args.output_jsonl)
    package_submission(args.output_jsonl, args.submission_zip)
    print(f"Saved predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()



