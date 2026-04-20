#!/bin/bash
# Launch 12 synth experiments in a dedicated tmux session.
# Each GPU runs 2 experiments SEQUENTIALLY (never more than 1 at a time per GPU).
# GPU 1 reserved (YOLO). GPU 7 kept free.
#
# Val sets:
#   Exp3 (full IAM + synth)  → val      (IAM original, 976 samples)
#   Exp4 (clean IAM + synth) → val_clean (IAM cleaned,  616 samples)
# Same val set across all scales within each group → fair cross-scale comparison.
#
# Usage: bash run_synth_experiments.sh

SESSION="synth_exp"
CONDA_ENV="ocr_IAM"
LMDB="./data/lmdb"
SYNTH_TRAIN="$LMDB/train_synth"
VAL_FULL="$LMDB/val"
VAL_CLEAN="$LMDB/val_clean"
LOG="./logs"
EPOCHS=150
LR=3e-4
HIDDEN=512
BS=64
WD=1e-4
DROPOUT=0.1

# Check prerequisites
for path in "$SYNTH_TRAIN" "$LMDB/train" "$LMDB/train_clean" "$VAL_FULL" "$VAL_CLEAN"; do
    if [ ! -f "$path/data.mdb" ]; then
        echo "ERROR: LMDB not ready: $path"
        exit 1
    fi
done

train_cmd() {
    local RUN=$1 MODE=$2 SYNTH_N=$3 GPU=$4
    local VAL
    [ "$MODE" = "clean" ] && VAL=$VAL_CLEAN || VAL=$VAL_FULL
    echo "conda run -n $CONDA_ENV --no-capture-output python -u train_iam.py \
        --mode $MODE --model v2 --run_name $RUN \
        --lmdb_dir $LMDB \
        --extra_train_lmdb $SYNTH_TRAIN \
        --synth_n $SYNTH_N \
        --val_lmdb $VAL \
        --epochs $EPOCHS --lr $LR --hidden $HIDDEN \
        --batch_size $BS --weight_decay $WD --dropout $DROPOUT \
        --gpu $GPU \
        2>&1 | tee $LOG/train_${RUN}.log"
}

# Kill existing session if any
tmux kill-session -t $SESSION 2>/dev/null || true

# Create session with monitor window
tmux new-session -d -s $SESSION -n "monitor"
tmux send-keys -t $SESSION:monitor \
    "watch -n 15 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv'" Enter

echo "=== Launching 12 experiments in tmux session '$SESSION' ==="
echo "    Each GPU runs 2 experiments sequentially (Exp3 then Exp4)"
echo ""

# Pairs: (gpu, scale, exp3_synth_n, exp4_synth_n)
# 6 scales × 2 experiments each (sequential) on GPUs 0-5. GPUs 6,7 free.
declare -a PAIRS=(
    "0  1x  6500   6500"
    "1  2x  13000  13000"
    "2  3x  19500  19500"
    "3  5x  32500  32500"
    "4  7x  45500  45500"
    "5  10x 65000  65000"
)

for PAIR in "${PAIRS[@]}"; do
    read -r GPU SCALE SYNTH3 SYNTH4 <<< "$PAIR"
    WIN="gpu${GPU}_${SCALE}"
    RUN3="exp3_full_${SCALE}"
    RUN4="exp4_clean_${SCALE}"

    CMD3=$(train_cmd $RUN3 original $SYNTH3 $GPU)
    CMD4=$(train_cmd $RUN4 clean    $SYNTH4 $GPU)

    tmux new-window -t $SESSION -n "$WIN"
    tmux send-keys -t $SESSION:"$WIN" "
cd /data02/home/tiger/thomas/final_project
echo '>>> GPU $GPU | Scale $SCALE | Starting $RUN3 ...'
$CMD3
echo '>>> GPU $GPU | Scale $SCALE | $RUN3 DONE. Starting $RUN4 ...'
$CMD4
echo '>>> GPU $GPU | Scale $SCALE | ALL DONE.'
" Enter

    echo "  GPU $GPU: $RUN3 → $RUN4  (sequential)"
done

echo ""
echo "Attach:       tmux attach -t $SESSION"
echo "List windows: tmux list-windows -t $SESSION"
echo "Monitor log:  tail -f logs/train_exp3_full_1x.log"
