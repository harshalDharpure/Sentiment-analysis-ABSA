# DimABSA 2026 - Competition Solution

**SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (Track A - Subtask 1)**

This repository contains a complete, competition-ready implementation for predicting Valence and Arousal scores for aspects in text.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download datasets:**
   ```bash
   python download_data.py
   ```

3. **Train model:**
   ```bash
   python -m src.train \
       --train-file data/raw/eng_laptop_train_alltasks.jsonl \
       --eval-file data/raw/eng_laptop_dev_task1.jsonl \
       --output-dir checkpoints \
       --batch-size 4 \
       --epochs 3
   ```

4. **Generate predictions:**
   ```bash
   python -m src.inference \
       --input-file data/raw/eng_laptop_dev_task1.jsonl \
       --checkpoint checkpoints/best_model.pt \
       --output-jsonl submission/predictions.jsonl \
       --submission-zip submission.zip
   ```

5. **Evaluate:**
   ```bash
   python metrics_subtask_1_2_3.py \
       -p submission/predictions.jsonl \
       -g data/raw/eng_laptop_dev_task1.jsonl \
       -t 1
   ```

## 📖 Full Documentation

See **[INSTRUCTIONS.md](INSTRUCTIONS.md)** for complete documentation, including:
- Detailed setup instructions
- Architecture overview
- Advanced usage (multi-domain training, hyperparameter tuning)
- Troubleshooting guide
- Tips for winning the competition

## 🏗️ Architecture

- **Model:** DeBERTa-v3-large with dual regression heads
- **Input:** Prompt-based formatting for better zero-shot performance
- **Output:** Valence and Arousal scores in [1.0, 9.0] range
- **Training:** MSE loss with Pearson correlation metrics

## 📊 Results

After training, you should see validation metrics:
- **PCC_V**: Pearson Correlation for Valence
- **PCC_A**: Pearson Correlation for Arousal
- **RMSE_VA**: Normalized Root Mean Squared Error

## 📝 License

This codebase is for SemEval-2026 Task 3 participation.

## 🔗 Links

- **Competition:** https://www.codabench.org/competitions/10918/
- **Repository:** https://github.com/DimABSA/DimABSA2026

