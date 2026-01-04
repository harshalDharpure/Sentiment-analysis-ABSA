# DimABSA 2026 Competition - Complete Instructions

## 🎯 Competition Overview

**SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (Track A - Subtask 1)**

**Goal:** Predict **Valence** and **Arousal** intensity scores (range 1.0-9.0) for specific aspects within text.

**Competition Link:** https://www.codabench.org/competitions/10918/

---

## 📁 Project Structure

```
dim_absa_2026/
├── data/
│   ├── raw/              # Downloaded datasets
│   ├── synthetic/         # Augmented data
│   └── processed/         # Processed datasets
├── src/
│   ├── config.py          # Configuration
│   ├── dataset.py         # Dataset loader with prompt formatting
│   ├── model.py           # DeBERTa-v3-large with dual regression heads
│   ├── train.py           # Training script
│   ├── inference.py       # Inference & submission generation
│   ├── augmentation.py    # Data augmentation utilities
│   └── utils.py           # Helper functions
├── submission/            # Generated predictions
├── checkpoints/           # Saved model checkpoints
├── download_data.py       # Download datasets
├── metrics_subtask_1_2_3.py  # Official evaluation script
├── requirements.txt       # Python dependencies
└── INSTRUCTIONS.md        # This file
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd dim_absa_2026
pip install -r requirements.txt
```

**Required packages:**
- `torch` (PyTorch)
- `transformers` (Hugging Face)
- `accelerate`
- `scikit-learn`
- `scipy`
- `nlpaug` (for augmentation)
- `numpy`
- `requests`

### Step 2: Download Datasets

```bash
python download_data.py
```

This downloads:
- Training data: `eng_laptop_train_alltasks.jsonl` and `eng_restaurant_train_alltasks.jsonl`
- Dev data: `eng_laptop_dev_task1.jsonl` and `eng_restaurant_dev_task1.jsonl`

**Manual download (alternative):**
- Repository: https://github.com/DimABSA/DimABSA2026
- Path: `task-dataset/track_a/subtask_1/eng/`

### Step 3: Train the Model

**Basic training:**
```bash
python -m src.train \
    --train-file data/raw/eng_laptop_train_alltasks.jsonl \
    --eval-file data/raw/eng_laptop_dev_task1.jsonl \
    --output-dir checkpoints \
    --batch-size 4 \
    --epochs 3 \
    --lr 2e-5
```

**With data augmentation:**
```bash
python -m src.train \
    --train-file data/raw/eng_laptop_train_alltasks.jsonl \
    --eval-file data/raw/eng_laptop_dev_task1.jsonl \
    --output-dir checkpoints \
    --batch-size 4 \
    --epochs 5 \
    --augment \
    --augment-ratio 0.3
```

**Full training options:**
```bash
python -m src.train \
    --train-file data/raw/eng_laptop_train_alltasks.jsonl \
    --eval-file data/raw/eng_laptop_dev_task1.jsonl \
    --output-dir checkpoints \
    --epochs 5 \
    --batch-size 4 \
    --eval-batch-size 8 \
    --lr 2e-5 \
    --max-length 256 \
    --warmup-ratio 0.1 \
    --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --seed 42 \
    --augment \
    --augment-ratio 0.3 \
    --pretrained microsoft/deberta-v3-large \
    --hidden-dim 512 \
    --dropout 0.1
```

**Training outputs:**
- Best model: `checkpoints/best_model.pt`
- Training logs: Console output with loss, Pearson correlation (Valence/Arousal), RMSE

### Step 4: Generate Predictions

```bash
python -m src.inference \
    --input-file data/raw/eng_laptop_dev_task1.jsonl \
    --checkpoint checkpoints/best_model.pt \
    --output-jsonl submission/predictions.jsonl \
    --submission-zip submission.zip \
    --batch-size 16
```

**Output format:**
```json
{"ID": "lap26_aspect_va_dev_1", "Aspect_VA": [{"Aspect": "touchscreen", "VA": "7.12#7.25"}]}
```

### Step 5: Evaluate Predictions

```bash
python metrics_subtask_1_2_3.py \
    -p submission/predictions.jsonl \
    -g data/raw/eng_laptop_dev_task1.jsonl \
    -t 1
```

**Metrics:**
- **PCC_V**: Pearson Correlation Coefficient for Valence
- **PCC_A**: Pearson Correlation Coefficient for Arousal  
- **RMSE_VA**: Root Mean Squared Error (normalized)

### Step 6: Submit to Codabench

1. **Prepare submission:**
   - The `submission.zip` file is automatically created by `inference.py`
   - Or manually zip the `submission/` folder

2. **Upload to Codabench:**
   - Go to: https://www.codabench.org/competitions/10918/
   - Click "My Submissions"
   - Select "Dev Phase" (999 submissions allowed) or "Test Phase" (3 submissions max)
   - Upload `submission.zip`

---

## 🏗️ Architecture Details

### Model Architecture

- **Backbone:** `microsoft/deberta-v3-large` (1.35B parameters)
- **Input:** Prompt-formatted text: `"Context: {text} | Analyze the aspect: {aspect} | Question: What is the Valence and Arousal intensity?"`
- **Head:** 
  - Shared projection: `768 → 512` (GELU + Dropout)
  - Dual regression heads: `512 → 1` each
  - Output scaling: Sigmoid → `[1.0, 9.0]` range

### Training Details

- **Loss:** `MSE(valence_pred, valence_true) + MSE(arousal_pred, arousal_true)`
- **Optimizer:** AdamW with weight decay
- **Scheduler:** Linear warmup + cosine decay
- **Gradient Clipping:** Max norm = 1.0
- **Validation Metrics:** Pearson Correlation (V & A), RMSE

### Data Format

**Training input (JSONL):**
```json
{"ID": "...", "Text": "...", "Quadruplet": [{"Aspect": "...", "VA": "7.12#7.12"}, ...]}
```

**Dev/Test input (JSONL):**
```json
{"ID": "...", "Text": "...", "Aspect": ["aspect1", "aspect2", ...]}
```

**Submission output (JSONL):**
```json
{"ID": "...", "Aspect_VA": [{"Aspect": "aspect1", "VA": "7.12#7.25"}, ...]}
```

---

## 🔧 Advanced Usage

### Multi-Domain Training

Train on multiple domains (laptop + restaurant):

```bash
# Combine datasets first (optional)
cat data/raw/eng_laptop_train_alltasks.jsonl data/raw/eng_restaurant_train_alltasks.jsonl > data/raw/combined_train.jsonl

python -m src.train \
    --train-file data/raw/combined_train.jsonl \
    --eval-file data/raw/eng_laptop_dev_task1.jsonl \
    --output-dir checkpoints \
    --batch-size 4 \
    --epochs 5
```

### Hyperparameter Tuning

**Learning rate sweep:**
```bash
for lr in 1e-5 2e-5 3e-5 5e-5; do
    python -m src.train \
        --train-file data/raw/eng_laptop_train_alltasks.jsonl \
        --eval-file data/raw/eng_laptop_dev_task1.jsonl \
        --output-dir checkpoints/lr_${lr} \
        --lr $lr \
        --epochs 3
done
```

**Batch size adjustment:**
- GPU memory permitting, increase `--batch-size` (e.g., 8, 16)
- Adjust `--lr` proportionally (e.g., batch_size=8 → lr=4e-5)

### Data Augmentation

**Current implementation:** Synonym replacement using `nlpaug`

**To use LLM-based augmentation:**
1. Edit `src/augmentation.py`
2. Replace `generate_synthetic()` with your LLM API call (GPT-4, Gemini, etc.)
3. Ensure output preserves original Valence/Arousal labels

### Model Checkpointing

**Resume training:**
```python
# Edit train.py to add --resume-from argument
# Load checkpoint: model.load_state_dict(torch.load(checkpoint_path))
```

**Ensemble predictions:**
```bash
# Train multiple models with different seeds
for seed in 42 123 456; do
    python -m src.train --seed $seed --output-dir checkpoints/seed_${seed} ...
done

# Average predictions in inference.py
```

---

## 📊 Evaluation Metrics

**Official metrics (Task 1):**
- **PCC_V**: Pearson Correlation for Valence (higher is better, max 1.0)
- **PCC_A**: Pearson Correlation for Arousal (higher is better, max 1.0)
- **RMSE_VA**: Normalized RMSE (lower is better)

**Target scores (competitive):**
- PCC_V > 0.85
- PCC_A > 0.85
- RMSE_VA < 0.15

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Solutions:**
- Reduce `--batch-size` (e.g., 2 or 4)
- Reduce `--max-length` (e.g., 128 or 192)
- Use gradient accumulation (add to `train.py`)
- Use mixed precision training (`torch.cuda.amp`)

### Low Validation Scores

**Check:**
1. Data format: Ensure training file has `Quadruplet` with `VA` field
2. Learning rate: Try different values (1e-5 to 5e-5)
3. Epochs: Train longer (5-10 epochs)
4. Model: Try `microsoft/deberta-v3-base` if large is too slow

### Submission Format Errors

**Ensure:**
- Output has `Aspect_VA` field (not `Aspect` or `Valence`/`Arousal`)
- VA format: `"7.12#7.25"` (not `7.12,7.25` or separate fields)
- Values in range [1.0, 9.0]
- All aspects from input are predicted

---

## 📚 References

- **Competition Repository:** https://github.com/DimABSA/DimABSA2026
- **Codabench Platform:** https://www.codabench.org/competitions/10918/
- **DeBERTa-v3 Paper:** https://arxiv.org/abs/2111.09543
- **ACOS Dataset:** https://github.com/NUSTM/ACOS

---

## 🎓 Tips for Winning

1. **Data Quality:**
   - Use both laptop and restaurant training data
   - Apply data augmentation (synonym replacement or LLM-based)
   - Filter out low-quality examples

2. **Model Improvements:**
   - Ensemble multiple models (different seeds/architectures)
   - Fine-tune on domain-specific data
   - Try larger models (DeBERTa-xlarge) if compute allows

3. **Training Strategy:**
   - Use learning rate scheduling (warmup + decay)
   - Early stopping based on validation PCC
   - Cross-validation for hyperparameter tuning

4. **Post-processing:**
   - Clamp predictions to [1.0, 9.0]
   - Handle edge cases (NULL aspects, implicit aspects)

5. **Submission:**
   - Test on dev set before test submission
   - Use all 3 test submissions wisely
   - Monitor leaderboard for insights

---

## 📝 License

This codebase is for SemEval-2026 Task 3 participation. Refer to the competition repository for dataset licenses.

---

## 🤝 Support

For competition-specific questions:
- **Competition Forum:** Check Codabench discussion board
- **GitHub Issues:** https://github.com/DimABSA/DimABSA2026/issues

For code issues:
- Review error messages and stack traces
- Check data format matches expected structure
- Verify all dependencies are installed correctly

---

**Good luck with the competition! 🏆**

