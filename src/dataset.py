"""
Dataset and prompt formatting utilities for Dimensional ABSA 2026.

Prompt format:

    "Context: {text} | Analyze the aspect: {aspect} | "
    "Question: What is the Valence and Arousal intensity?"

This module provides:
    - `format_input`: prompt construction
    - `DimABSADataset`: torch Dataset that expands multi-aspect rows
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

PROMPT_TEMPLATE = (
    "Context: {text} | Analyze the aspect: {aspect} | "
    "Question: What is the Valence and Arousal intensity?"
)


def format_input(sample: Dict[str, str]) -> str:
    """
    Format a single training/example instance into a natural-language prompt.
    """
    text = sample.get("text", "")
    aspect = sample.get("aspect", "")
    return PROMPT_TEMPLATE.format(text=text, aspect=aspect)


@dataclass
class DimABSAExample:
    """
    In-memory representation of a single (text, aspect) pair with optional labels.
    """

    example_id: str
    text: str
    aspect: str
    valence: Optional[float] = None
    arousal: Optional[float] = None


def _normalize_label(value: Optional[Sequence[float] | float], idx: int) -> Optional[float]:
    """
    Support labels as either scalar or list; return the idx-th value when list.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return float(value[idx % len(value)])
    return float(value)


class DimABSADataset(Dataset):
    """
    Torch dataset for prompt-based Dimensional ABSA.

    - Expands rows with multiple aspects into multiple examples.
    - Handles optional valence/arousal labels.
    - Tokenizes prompts to fixed max_length for simple collation.
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        add_special_tokens: bool = True,
        use_padding: bool = True,
        include_labels: bool = True,
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.use_padding = use_padding
        self.include_labels = include_labels

        self.examples: List[DimABSAExample] = []
        self._load()

    def _load(self) -> None:
        """
        Load data from JSONL file. Supports two formats:
        1. Training format: {"Quadruplet": [{"Aspect": "...", "VA": "v#a"}, ...]}
        2. Dev/Test format: {"Aspect": ["aspect1", "aspect2", ...]}
        """
        with self.file_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                text = record.get("Text", "")
                ex_id = record.get("ID", "")

                # Check for training format (Quadruplet with VA)
                quadruplets = record.get("Quadruplet", [])
                if quadruplets and self.include_labels:
                    # Training data: extract from Quadruplet
                    for quad in quadruplets:
                        aspect = quad.get("Aspect", "")
                        if aspect == "NULL" or not aspect:
                            continue
                        va_str = quad.get("VA", "")
                        if va_str and "#" in va_str:
                            try:
                                v_str, a_str = va_str.split("#")
                                v_label = float(v_str.strip())
                                a_label = float(a_str.strip())
                                self.examples.append(
                                    DimABSAExample(
                                        example_id=f"{ex_id}__{aspect}",
                                        text=text,
                                        aspect=aspect,
                                        valence=v_label,
                                        arousal=a_label,
                                    )
                                )
                            except (ValueError, AttributeError):
                                continue
                else:
                    # Dev/Test format: just Aspect list
                    aspects = record.get("Aspect", []) or []
                    if isinstance(aspects, str):
                        aspects = [aspects]
                    
                    for idx, aspect in enumerate(aspects):
                        if not aspect or aspect == "NULL":
                            continue
                        v_label = None
                        a_label = None
                        if self.include_labels:
                            # Try to find matching Quadruplet if available
                            quadruplets = record.get("Quadruplet", [])
                            for quad in quadruplets:
                                if quad.get("Aspect", "").lower() == aspect.lower():
                                    va_str = quad.get("VA", "")
                                    if va_str and "#" in va_str:
                                        try:
                                            v_str, a_str = va_str.split("#")
                                            v_label = float(v_str.strip())
                                            a_label = float(a_str.strip())
                                        except (ValueError, AttributeError):
                                            pass
                                    break
                        
                        if self.include_labels and (v_label is None or a_label is None):
                            continue
                        
                        self.examples.append(
                            DimABSAExample(
                                example_id=f"{ex_id}__{idx}",
                                text=text,
                                aspect=aspect,
                                valence=v_label,
                                arousal=a_label,
                            )
                        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | float | None]:
        example = self.examples[idx]
        prompt = format_input({"text": example.text, "aspect": example.aspect})

        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length" if self.use_padding else False,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

        item: Dict[str, torch.Tensor | str | float | None] = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "id": example.example_id,
            "text": example.text,
            "aspect": example.aspect,
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        if self.include_labels:
            item["valence"] = torch.tensor(example.valence, dtype=torch.float32) if example.valence is not None else None
            item["arousal"] = torch.tensor(example.arousal, dtype=torch.float32) if example.arousal is not None else None

        return item


