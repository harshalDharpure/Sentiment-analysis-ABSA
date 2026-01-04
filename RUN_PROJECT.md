# How to Run the Project and Get Accuracy Scores

This guide will walk you through running the DimABSA 2026 project and obtaining accuracy metrics.

## Step-by-Step Instructions

### Step 1: Install Dependencies

First, navigate to the project directory and install required packages:

```bash
cd dim_absa_2026
pip install -r requirements.txt
```

### Step 2: Train the Model

Train the model using the training data:

```bash
python -m src.train \
    --train-file data/raw/eng_laptop_train_alltasks.jsonl \
    --eval-file data/raw/eng_laptop_dev_task1.jsonl \
    --output-dir checkpoints \
    --batch-size 4 \
    --epochs 3 \
    --lr 2e-5
```

**Note:** Training may take some time depending on your hardware. The model will be saved to `checkpoints/best_model.pt`.

### Step 3: Generate Predictions

After training, generate predictions on the dev set:

```bash
python -m src.inference \
    --input-file data/raw/eng_laptop_dev_task1.jsonl \
    --checkpoint checkpoints/best_model.pt \
    --output-jsonl submission/predictions.jsonl \
    --submission-zip submission.zip \
    --batch-size 16
```

### Step 4: Evaluate and Get Accuracy Scores

Run the evaluation script to get the accuracy metrics:

```bash
python metrics_subtask_1_2_3.py \
    -p submission/predictions.jsonl \
    -g data/raw/eng_laptop_dev_task1.jsonl \
    -t 1
```

## Understanding the Accuracy Metrics

The evaluation script outputs three main metrics:

1. **PCC_V** (Pearson Correlation Coefficient for Valence)
   - Range: -1.0 to 1.0 (higher is better)
   - Measures how well predicted Valence scores correlate with ground truth

2. **PCC_A** (Pearson Correlation Coefficient for Arousal)
   - Range: -1.0 to 1.0 (higher is better)
   - Measures how well predicted Arousal scores correlate with ground truth

3. **RMSE_VA** (Root Mean Squared Error, normalized)
   - Lower is better
   - Measures the average prediction error for both Valence and Arousal

**Target Scores (Competitive):**
- PCC_V > 0.85
- PCC_A > 0.85
- RMSE_VA < 0.15

## Quick Run Script (All Steps)

You can also run all steps in sequence. Here's a complete example:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model
python -m src.train \
    --train-file data/raw/eng_laptop_train_alltasks.jsonl \
    --eval-file data/raw/eng_laptop_dev_task1.jsonl \
    --output-dir checkpoints \
    --batch-size 4 \
    --epochs 3

# 3. Generate predictions
python -m src.inference \
    --input-file data/raw/eng_laptop_dev_task1.jsonl \
    --checkpoint checkpoints/best_model.pt \
    --output-jsonl submission/predictions.jsonl \
    --submission-zip submission.zip

# 4. Evaluate
python metrics_subtask_1_2_3.py \
    -p submission/predictions.jsonl \
    -g data/raw/eng_laptop_dev_task1.jsonl \
    -t 1
```

## Troubleshooting

- **Out of Memory**: Reduce `--batch-size` to 2 or use a smaller model
- **Missing Checkpoint**: Make sure training completed successfully and `checkpoints/best_model.pt` exists
- **Missing Predictions**: Ensure the inference step completed and `submission/predictions.jsonl` exists

## Alternative: Using Restaurant Domain

You can also train and evaluate on the restaurant domain:

```bash
# Train on restaurant data
python -m src.train \
    --train-file data/raw/eng_restaurant_train_alltasks.jsonl \
    --eval-file data/raw/eng_restaurant_dev_task1.jsonl \
    --output-dir checkpoints \
    --batch-size 4 \
    --epochs 3

# Generate predictions
python -m src.inference \
    --input-file data/raw/eng_restaurant_dev_task1.jsonl \
    --checkpoint checkpoints/best_model.pt \
    --output-jsonl submission/predictions.jsonl \
    --submission-zip submission.zip

# Evaluate
python metrics_subtask_1_2_3.py \
    -p submission/predictions.jsonl \
    -g data/raw/eng_restaurant_dev_task1.jsonl \
    -t 1
```

