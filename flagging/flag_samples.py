"""
CER-based flagging pipeline.

Strategy (three-level funnel):
  1. Compute CER(claude_pred, ground_truth) and CER(gpt4o_pred, ground_truth)
  2. Flag a sample if BOTH VLMs exceed the CER threshold
  3. Flagged samples go to manual review

Usage:
    python flag_samples.py \
        --claude_results results/claude_test.json \
        --gpt4o_results  results/gpt4o_test.json \
        --output_dir     results/flagging \
        --cer_threshold  0.3
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from flagging.cer import compute_cer

# Error type labels for manual review
ERROR_TYPES = [
    'mislabeled',        # outright wrong transcription in ground truth
    'crossed_out',       # word(s) crossed out in image
    'missing_words',     # words present in image but absent from label
    'ambiguous',         # genuinely ambiguous handwriting
    'image_quality',     # degraded / low-quality scan
    'ok',                # VLMs were wrong, ground truth is correct
]


def load_results(path):
    with open(path) as f:
        return json.load(f)


def merge_results(claude_results, gpt4o_results):
    """
    Merge Claude and GPT-4o results by line ID.
    Returns list of dicts with both predictions.
    """
    merged = []
    all_ids = set(claude_results.keys()) | set(gpt4o_results.keys())
    for line_id in sorted(all_ids):
        c = claude_results.get(line_id, {})
        g = gpt4o_results.get(line_id, {})
        gt = c.get('ground_truth') or g.get('ground_truth', '')
        merged.append({
            'id': line_id,
            'ground_truth': gt,
            'claude_pred': c.get('claude_pred', ''),
            'gpt4o_pred': g.get('gpt4o_pred', ''),
            'image_path': c.get('image_path') or g.get('image_path', ''),
            'writer': c.get('writer') or g.get('writer', ''),
        })
    return merged


def flag_samples(merged, cer_threshold=0.3):
    """
    Flag samples where BOTH VLMs exceed cer_threshold.
    Returns two lists: flagged, unflagged.
    """
    flagged = []
    unflagged = []

    for rec in merged:
        gt = rec['ground_truth']
        cer_claude = compute_cer(rec['claude_pred'], gt) if rec['claude_pred'] else 1.0
        cer_gpt4o  = compute_cer(rec['gpt4o_pred'], gt)  if rec['gpt4o_pred']  else 1.0

        rec['cer_claude'] = round(cer_claude, 4)
        rec['cer_gpt4o']  = round(cer_gpt4o, 4)
        rec['cer_avg']    = round((cer_claude + cer_gpt4o) / 2, 4)
        rec['flagged']    = (cer_claude >= cer_threshold) and (cer_gpt4o >= cer_threshold)

        if rec['flagged']:
            rec['error_type'] = ''  # to be filled in manual review
            flagged.append(rec)
        else:
            unflagged.append(rec)

    return flagged, unflagged


def save_outputs(flagged, unflagged, output_dir, cer_threshold):
    os.makedirs(output_dir, exist_ok=True)

    # Full flagged list for manual review (CSV for easy editing)
    df_flag = pd.DataFrame(flagged)
    flag_csv = os.path.join(output_dir, 'flagged_for_review.csv')
    df_flag.to_csv(flag_csv, index=False)
    print(f"Flagged samples CSV: {flag_csv}")

    # Just the IDs (for prepare_lmdb.py --flagged_ids)
    id_file = os.path.join(output_dir, 'flagged_ids.txt')
    with open(id_file, 'w') as f:
        for rec in flagged:
            f.write(rec['id'] + '\n')
    print(f"Flagged IDs file: {id_file}")

    # Full results (all samples with CER scores)
    all_samples = flagged + unflagged
    df_all = pd.DataFrame(all_samples)
    all_csv = os.path.join(output_dir, 'all_results.csv')
    df_all.to_csv(all_csv, index=False)
    print(f"All results CSV: {all_csv}")

    # Summary
    summary = {
        'total_samples': len(all_samples),
        'flagged_count': len(flagged),
        'flagged_rate': round(len(flagged) / max(len(all_samples), 1), 4),
        'cer_threshold': cer_threshold,
        'mean_cer_claude': round(df_all['cer_claude'].mean(), 4),
        'mean_cer_gpt4o':  round(df_all['cer_gpt4o'].mean(), 4),
        'flagged_by_writer': (
            df_flag.groupby('writer').size().to_dict() if not df_flag.empty else {}
        ),
    }
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")
    print(f"\n{'='*50}")
    print(f"Total samples:  {summary['total_samples']}")
    print(f"Flagged:        {summary['flagged_count']} ({summary['flagged_rate']*100:.1f}%)")
    print(f"CER threshold:  {cer_threshold}")
    print(f"Mean CER Claude: {summary['mean_cer_claude']:.4f}")
    print(f"Mean CER GPT-4o: {summary['mean_cer_gpt4o']:.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--claude_results', required=True)
    parser.add_argument('--gpt4o_results', required=True)
    parser.add_argument('--output_dir', default='results/flagging')
    parser.add_argument('--cer_threshold', type=float, default=0.3,
                        help='CER threshold; samples above this for BOTH VLMs are flagged')
    args = parser.parse_args()

    print("Loading VLM results...")
    claude_res = load_results(args.claude_results)
    gpt4o_res  = load_results(args.gpt4o_results)
    print(f"  Claude: {len(claude_res)} | GPT-4o: {len(gpt4o_res)}")

    merged = merge_results(claude_res, gpt4o_res)
    flagged, unflagged = flag_samples(merged, args.cer_threshold)
    save_outputs(flagged, unflagged, args.output_dir, args.cer_threshold)


if __name__ == '__main__':
    main()
