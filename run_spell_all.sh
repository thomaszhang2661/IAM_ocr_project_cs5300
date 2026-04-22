#!/bin/bash
# Run spell correction evaluation for all experiments in parallel across 5 GPUs
# Usage: nohup bash run_spell_all.sh > results/spell_all_master.log 2>&1 &

BASE=/data02/home/tiger/thomas/final_project
cd $BASE

CONF3=results/model_confusion_exp3.csv
CONF5=results/model_confusion_exp5.csv
LM=lm/word_4gram.arpa

# Map: "exp_dir|confusion_csv|out_name"
declare -a EXPS=(
  "exp1_full_iam|$CONF3|beam_spell_exp1_full_iam"
  "exp2_clean_iam|$CONF3|beam_spell_exp2_clean_iam"
  "exp3_full_1x|$CONF3|beam_spell_exp3_full_1x"
  "exp3_full_2x|$CONF3|beam_spell_exp3_full_2x"
  "exp3_full_3x|$CONF3|beam_spell_exp3_full_3x"
  "exp3_full_5x|$CONF3|beam_spell_exp3_full_5x"
  "exp3_full_7x|$CONF3|beam_spell_exp3_full_7x"
  "exp4_clean_1x|$CONF3|beam_spell_exp4_clean_1x"
  "exp4_clean_2x|$CONF3|beam_spell_exp4_clean_2x"
  "exp4_clean_3x|$CONF3|beam_spell_exp4_clean_3x"
  "exp4_clean_5x|$CONF3|beam_spell_exp4_clean_5x"
  "exp4_clean_7x|$CONF3|beam_spell_exp4_clean_7x"
  "exp4_clean_10x|$CONF3|beam_spell_exp4_clean_10x"
  "exp5_clean2_base|$CONF5|beam_spell_exp5_clean2_base"
  "exp5_clean2_1x|$CONF5|beam_spell_exp5_clean2_1x"
  "exp5_clean2_2x|$CONF5|beam_spell_exp5_clean2_2x"
  "exp5_clean2_3x|$CONF5|beam_spell_exp5_clean2_3x"
  "exp5_clean2_5x|$CONF5|beam_spell_exp5_clean2_5x"
  "exp5_clean2_7x|$CONF5|beam_spell_exp5_clean2_7x"
)

GPU=0
PIDS=()

for entry in "${EXPS[@]}"; do
  IFS='|' read -r exp_dir conf_csv out_name <<< "$entry"
  ckpt="checkpoints/${exp_dir}/best.pt"
  out_json="results/${out_name}.json"
  log="results/${out_name}.log"

  if [ -f "$out_json" ]; then
    echo "SKIP (exists): $out_json"
    continue
  fi

  echo "Launching: $exp_dir → GPU $GPU"
  conda run -n ocr_IAM python eval_spell_corrector.py \
    --checkpoint "$ckpt" \
    --confusion_csv "$conf_csv" \
    --confusion_min 5 \
    --run_name "$exp_dir" \
    --lm_path "$LM" \
    --gpu $GPU \
    --out_json "$out_json" \
    > "$log" 2>&1 &

  PIDS+=($!)
  GPU=$(( (GPU + 1) % 5 ))
  sleep 0.5
done

echo "Launched ${#PIDS[@]} jobs (PIDs: ${PIDS[*]}). Waiting..."
wait
echo "All done."

# Print summary
echo ""
echo "=== Results Summary ==="
for entry in "${EXPS[@]}"; do
  IFS='|' read -r exp_dir conf_csv out_name <<< "$entry"
  out_json="results/${out_name}.json"
  if [ -f "$out_json" ]; then
    python3 -c "
import json
d = json.load(open('$out_json'))
print(f\"  $exp_dir: greedy={d['greedy_cer']:.2f}% beam={d['beam_cer']:.2f}% spell={d['spell_cer']:.2f}%\")
" 2>/dev/null
  else
    echo "  $exp_dir: MISSING (check results/${out_name}.log)"
  fi
done
