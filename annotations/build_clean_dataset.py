"""
Compute final_annotation from the human-reviewed audit CSV,
then build clean train/test datasets for CRNN training.

final_annotation logic:
  - human_verdict = confirmed_correct → keep ground_truth
  - human_verdict = confirmed_error, human_corrected filled → use human_corrected
  - human_verdict = confirmed_error, human_corrected empty  → use doubao_corrected
  - anything else (skip / not reviewed)                     → keep ground_truth

Output:
  annotations/final_train.csv   — (idx, final_annotation) for training
  annotations/final_test.csv    — (idx, final_annotation) for evaluation
  annotations/correction_stats.json — how many samples were corrected

Usage:
    python annotations/build_clean_dataset.py \
        --audit_train annotations/audit_train.csv \
        --audit_test  annotations/audit_test.csv \
        --output_dir  annotations/
"""

import argparse
import json
import os
import pandas as pd


def compute_final_annotation(row) -> str:
    gt        = str(row.get('ground_truth',   '') or '')
    verdict   = str(row.get('human_verdict',  '') or '').strip().lower()
    h_correct = str(row.get('human_corrected','') or '').strip()
    d_correct = str(row.get('doubao_corrected','') or '').strip()

    if verdict == 'confirmed_error':
        if h_correct:
            return h_correct       # human typed correction takes priority
        elif d_correct:
            return d_correct       # accept model suggestion
        else:
            return gt              # no correction available, keep original
    else:
        return gt                  # confirmed_correct / skip / empty → keep original


def process_split(audit_csv: str, split_name: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(audit_csv)
    df['final_annotation'] = df.apply(compute_final_annotation, axis=1)

    n_total     = len(df)
    n_reviewed  = (df['human_verdict'].notna() & (df['human_verdict'] != '')).sum()
    n_corrected = (df['human_verdict'].str.strip().str.lower() == 'confirmed_error').sum()
    n_model_fix = (
        (df['human_verdict'].str.strip().str.lower() == 'confirmed_error') &
        (df['human_corrected'].isna() | (df['human_corrected'] == ''))
    ).sum()
    n_human_fix = (
        (df['human_verdict'].str.strip().str.lower() == 'confirmed_error') &
        df['human_corrected'].notna() & (df['human_corrected'] != '')
    ).sum()

    stats = {
        'split':         split_name,
        'total':         int(n_total),
        'reviewed':      int(n_reviewed),
        'corrected':     int(n_corrected),
        'model_fix':     int(n_model_fix),   # accepted doubao suggestion
        'human_fix':     int(n_human_fix),   # human typed their own correction
        'unchanged':     int(n_total - n_corrected),
    }
    return df[['idx', 'split', 'ground_truth', 'final_annotation']], stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit_train', default=None,
                        help='Reviewed audit CSV for train split')
    parser.add_argument('--audit_test',  default=None,
                        help='Reviewed audit CSV for test split')
    parser.add_argument('--output_dir',  default='annotations/')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_stats = []

    for split_name, audit_csv in [('train', args.audit_train),
                                   ('test',  args.audit_test)]:
        if not audit_csv or not os.path.exists(audit_csv):
            print(f'Skip {split_name}: no audit file')
            continue

        df_final, stats = process_split(audit_csv, split_name)
        out_path = os.path.join(args.output_dir, f'final_{split_name}.csv')
        df_final.to_csv(out_path, index=False)
        all_stats.append(stats)

        print(f'\n{split_name.upper()} split:')
        print(f'  Total:    {stats["total"]}')
        print(f'  Reviewed: {stats["reviewed"]}')
        print(f'  Corrected:{stats["corrected"]}  '
              f'({stats["model_fix"]} model, {stats["human_fix"]} human)')
        print(f'  Unchanged:{stats["unchanged"]}')
        print(f'  → {out_path}')

    stats_path = os.path.join(args.output_dir, 'correction_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f'\nStats: {stats_path}')


if __name__ == '__main__':
    main()
