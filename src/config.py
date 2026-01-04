"""
Configuration and hyperparameters for Dimensional ABSA 2026 Track A.

These are lightweight placeholders; feel free to override via argparse or
environment variables from `train.py` and `inference.py`.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    pretrained_model_name: str = "microsoft/deberta-v3-large"
    hidden_dim: int = 512
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_seq_length: int = 256


@dataclass
class PathsConfig:
    data_root: str = "dim_absa_2026/data"
    raw_dir: str = "dim_absa_2026/data/raw"
    synthetic_dir: str = "dim_absa_2026/data/synthetic"
    processed_dir: str = "dim_absa_2026/data/processed"
    submission_dir: str = "dim_absa_2026/submission"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def get_default_config() -> Config:
    """Return a default configuration object."""
    return Config()


