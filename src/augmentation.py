"""
Synthetic data generation / augmentation utilities for Dimensional ABSA 2026.

The goal is to allow simple augmentation while keeping the interface
compatible with a future LLM-based rewriting function.
"""

from __future__ import annotations

from typing import Tuple

try:
    import nlpaug.augmenter.word as naw
except ImportError:  # pragma: no cover
    naw = None  # soft dependency; handled gracefully below


def build_augmenter():
    """
    Create a synonym replacement augmenter.
    Returns None if nlpaug is not installed.
    """
    if naw is None:
        return None
    try:
        return naw.SynonymAug(aug_src="wordnet")
    except Exception:
        return None


def generate_synthetic(
    text: str,
    aspect: str,
    valence: float,
    arousal: float,
) -> Tuple[str, str, float, float]:
    """
    Generate a synthetic variant of the input while preserving labels.

    If `nlpaug` is available, performs synonym replacement; otherwise
    returns the original text.
    """
    augmenter = build_augmenter()
    if augmenter is None:
        return text, aspect, valence, arousal

    try:
        augmented_text = augmenter.augment(text)
    except Exception:
        augmented_text = text

    return augmented_text, aspect, valence, arousal



