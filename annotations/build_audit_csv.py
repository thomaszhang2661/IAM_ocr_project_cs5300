"""
Merge Doubao VLM results into a full audit CSV with human-review columns.

Run this AFTER doubao_check.py finishes to produce the audit file
you fill in manually (human_verdict, human_corrected).

Usage:
    python annotations/build_audit_csv.py \
        --vlm_csv  results/doubao_test_all.csv \
        --output   annotations/audit_test.csv

Then open audit_test.csv in Excel / Numbers, fill in:
  human_verdict  : confirmed_correct | confirmed_error | skip
  human_corrected: (only if doubao_corrected is wrong — leave blank otherwise)

Then run build_clean_dataset.py to compute final_annotation.
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm_csv', required=True,
                        help='doubao_<split>_all.csv from doubao_check.py')
    parser.add_argument('--output',  required=True,
                        help='Output audit CSV path')
    args = parser.parse_args()

    df = pd.read_csv(args.vlm_csv)

    # Keep only relevant columns, rename for clarity
    cols = {
        'idx':            'idx',
        'split':          'split',
        'ground_truth':   'ground_truth',
        'verdict':        'doubao_verdict',
        'reason':         'doubao_reason',
        'corrected_text': 'doubao_corrected',
    }
    df = df.rename(columns={v: k for k, v in {
        'verdict':        'doubao_verdict',
        'reason':         'doubao_reason',
        'corrected_text': 'doubao_corrected',
    }.items()})

    # Add human review columns (blank — to be filled in manually)
    df['human_verdict']   = ''   # confirmed_correct | confirmed_error | skip
    df['human_corrected'] = ''   # fill only if doubao_corrected is wrong

    # Sort: INCORRECT first (highest priority for review), then AMBIGUOUS, then CORRECT
    order = {'INCORRECT': 0, 'AMBIGUOUS': 1, 'CORRECT': 2}
    df['_sort'] = df['doubao_verdict'].map(order).fillna(3)
    df = df.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)

    n_incorrect = (df['doubao_verdict'] == 'INCORRECT').sum()
    n_ambiguous = (df['doubao_verdict'] == 'AMBIGUOUS').sum()
    n_correct   = (df['doubao_verdict'] == 'CORRECT').sum()

    print(f'Audit CSV: {args.output}')
    print(f'  Total:     {len(df)}')
    print(f'  INCORRECT: {n_incorrect}  ← review these first')
    print(f'  AMBIGUOUS: {n_ambiguous}  ← review if time allows')
    print(f'  CORRECT:   {n_correct}   ← skip')
    print(f'\nOpen {args.output} and fill in human_verdict + human_corrected columns.')


if __name__ == '__main__':
    main()
