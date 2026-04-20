#!/usr/bin/env bash
# ============================================================
# CS5300 Final Project: End-to-End Pipeline
# VLM-based Annotation Quality Analysis of IAM
# ============================================================
# Usage: edit the variables below then run:
#   bash run_pipeline.sh
# ============================================================

set -e

# ---- CONFIG ------------------------------------------------
IAM_ROOT="/path/to/iam"                         # <-- change this
OCR_REPO="/Users/zhangjian/Documents/up366ocr-f7c7d279999b779b27f74584ac3e5f0fa191a051-f7c7d279999b779b27f74584ac3e5f0fa191a051"
CFG="configs/config_iam.yaml"

ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"         # set in env
OPENAI_API_KEY="${OPENAI_API_KEY}"               # set in env

SAMPLE_N=500       # number of lines to sample for VLM inference (test split)
CER_THRESHOLD=0.3  # flag if both VLMs exceed this CER

# ---- STEP 1: VLM INFERENCE ---------------------------------
echo ""
echo "=== STEP 1: VLM Inference ==="
python vlm_inference/claude_inference.py \
    --iam_root "$IAM_ROOT" \
    --split test \
    --output results/claude_test.json \
    --sample $SAMPLE_N

python vlm_inference/gpt4o_inference.py \
    --iam_root "$IAM_ROOT" \
    --split test \
    --output results/gpt4o_test.json \
    --sample $SAMPLE_N

# ---- STEP 2: CER FLAGGING ----------------------------------
echo ""
echo "=== STEP 2: CER-based Flagging ==="
python flagging/flag_samples.py \
    --claude_results results/claude_test.json \
    --gpt4o_results  results/gpt4o_test.json \
    --output_dir     results/flagging \
    --cer_threshold  $CER_THRESHOLD

echo ""
echo ">>> MANUAL STEP: Open results/flagging/flagged_for_review.csv"
echo ">>> Fill in the 'error_type' column for each flagged sample."
echo ">>> Error types: mislabeled | crossed_out | missing_words | ambiguous | image_quality | ok"
echo ">>> Press ENTER when done..."
read -r

# ---- STEP 3: ERROR ANALYSIS --------------------------------
echo ""
echo "=== STEP 3: Error Analysis ==="
python flagging/analyze_errors.py \
    --flagged_csv results/flagging/flagged_for_review.csv \
    --all_csv     results/flagging/all_results.csv \
    --output_dir  results/analysis

# ---- STEP 4: TRAIN ORIGINAL --------------------------------
echo ""
echo "=== STEP 4: Train CRNN (Original IAM) ==="
PYTHONPATH="$OCR_REPO:$PYTHONPATH" python train_iam.py \
    --cfg_file  "$CFG" \
    --iam_root  "$IAM_ROOT" \
    --mode      original \
    --run_name  orig \
    --ocr_repo  "$OCR_REPO"

# ---- STEP 5: TRAIN CLEAN -----------------------------------
echo ""
echo "=== STEP 5: Train CRNN (Clean IAM, flagged removed) ==="
PYTHONPATH="$OCR_REPO:$PYTHONPATH" python train_iam.py \
    --cfg_file   "$CFG" \
    --iam_root   "$IAM_ROOT" \
    --mode       clean \
    --flagged_ids results/flagging/flagged_ids.txt \
    --run_name   clean \
    --ocr_repo   "$OCR_REPO"

# ---- STEP 6: EVALUATE BOTH ---------------------------------
echo ""
echo "=== STEP 6: Evaluate on Test Split ==="
PYTHONPATH="$OCR_REPO:$PYTHONPATH" python evaluate_iam.py \
    --cfg_file   "$CFG" \
    --iam_root   "$IAM_ROOT" \
    --checkpoint checkpoints/orig/best.pt \
    --run_name   orig \
    --ocr_repo   "$OCR_REPO"

PYTHONPATH="$OCR_REPO:$PYTHONPATH" python evaluate_iam.py \
    --cfg_file   "$CFG" \
    --iam_root   "$IAM_ROOT" \
    --checkpoint checkpoints/clean/best.pt \
    --run_name   clean \
    --ocr_repo   "$OCR_REPO"

echo ""
echo "=== DONE ==="
echo "Results: results/test_results_orig.json"
echo "         results/test_results_clean.json"
