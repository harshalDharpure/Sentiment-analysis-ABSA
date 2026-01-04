#!/usr/bin/env bash
set -euo pipefail

# Simple placeholder pipeline script for DimABSA 2026 Track A.
# To be expanded with full train / inference / submission steps.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

echo "[1/3] Downloading example data..."
python download_data.py

echo "[2/3] Training"
echo "Provide a labeled train file. Example:"
echo "python -m src.train --train-file data/raw/eng_laptop_train_task1.jsonl --eval-file data/raw/eng_laptop_dev_task1.jsonl --output-dir checkpoints"

echo "[3/3] Inference / submission"
echo "After training, run:"
echo "python -m src.inference --input-file data/raw/eng_laptop_dev_task1.jsonl --checkpoint checkpoints/best_model.pt --output-jsonl submission/predictions.jsonl --submission-zip submission.zip"

echo "Pipeline scaffold completed."


