import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DimABSAModel(nn.Module):
    """
    DeBERTa-v3-based backbone with a shared projection layer and
    two separate regression heads for Valence and Arousal.

    By default, uses `microsoft/deberta-v3-large` as the backbone.
    """

    def __init__(
        self,
        pretrained_model_name: str = "microsoft/deberta-v3-large",
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Load backbone configuration & model
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.backbone = AutoModel.from_pretrained(pretrained_model_name, config=self.config, torch_dtype=torch.float32)

        backbone_hidden_size = self.config.hidden_size

        # Shared projection from backbone hidden size -> hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate regression heads for Valence and Arousal
        # Output range: [1.0, 9.0] using sigmoid scaling
        self.valence_head = nn.Linear(hidden_dim, 1)
        self.arousal_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass through the backbone and dual regression heads.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (optional, kept for API compatibility)

        Returns:
            dict with:
                - valence: [batch_size, 1]
                - arousal: [batch_size, 1]
                - pooled_output: [batch_size, hidden_dim] (shared representation)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # Standard transformer-style pooling: use the first token ([CLS]-like) representation
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state
        cls_embedding = sequence_output[:, 0, :]  # [batch_size, hidden_size]

        shared_repr = self.projection(cls_embedding)  # [batch_size, hidden_dim]

        valence_raw = self.valence_head(shared_repr)
        arousal_raw = self.arousal_head(shared_repr)
        
        # Scale to [1.0, 9.0] range using sigmoid: sigmoid(x) * 8 + 1
        # This ensures outputs are always in valid range for evaluation
        valence = torch.sigmoid(valence_raw) * 8.0 + 1.0
        arousal = torch.sigmoid(arousal_raw) * 8.0 + 1.0

        return {
            "valence": valence,
            "arousal": arousal,
            "pooled_output": shared_repr,
        }


def load_dim_absa_model_from_pretrained(
    checkpoint_path: str | None = None,
    pretrained_model_name: str = "microsoft/deberta-v3-large",
    device: str | torch.device = "cpu",
    **kwargs,
) -> DimABSAModel:
    """
    Utility to instantiate the model and optionally load a fine-tuned checkpoint.

    Args:
        checkpoint_path: path to a `.pt` / `.bin` state_dict file (optional).
        pretrained_model_name: backbone model name.
        device: torch device for the returned model.
        **kwargs: forwarded to `DimABSAModel` (e.g. hidden_dim, dropout).
    """
    model = DimABSAModel(
        pretrained_model_name=pretrained_model_name,
        **kwargs,
    )

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

    model.to(device)
    return model


