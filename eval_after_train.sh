#!/bin/bash
# Wait for training log to show "Epoch 150" final completion, then eval

MODEL=$1
GPU=$2
CKPT="checkpoints/${MODEL}/best.pt"
OUT="results/test_results_${MODEL}.json"

echo "[eval_after_train] Waiting for ${MODEL} to finish training..."
while true; do
    if grep -q "Epoch 150.*val_CER" "logs/train_${MODEL}.log" 2>/dev/null; then
        echo "[eval_after_train] Training done. Running eval..."
        break
    fi
    sleep 30
done

# Greedy eval
conda run -n ocr_IAM python evaluate_iam.py \
    --checkpoint "${CKPT}" --split test --model v2 --hidden 512 \
    --run_name "${MODEL}" --gpu "${GPU}" 2>&1 | tee "logs/eval_${MODEL}.log"

# Beam v3
conda run -n ocr_IAM python decode_beam_v2.py \
    --checkpoint "${CKPT}" --split test \
    --lm_path lm/word_4gram.arpa \
    --confusion_min 999999 \
    --out_json "results/beam_v3_${MODEL}.json" \
    --gpu "${GPU}" 2>&1 | tee "logs/beam_v3_${MODEL}.log"

echo "[eval_after_train] Done: ${MODEL}"
