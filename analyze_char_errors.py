"""
Per-character error rate analysis.
Compares two checkpoints on the test set and outputs:
  - per-character substitution / deletion / insertion counts
  - per-character CER
  - comparison table (checkpoint A vs B)

Usage:
    python analyze_char_errors.py \
        --ckpt_a checkpoints/exp1_full_iam/best.pt \
        --ckpt_b checkpoints/exp3_full_10x/best.pt \
        --name_a exp1_full_iam --name_b exp3_full_10x \
        --out_csv results/char_errors.csv
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from htr_model.dataset import LMDBDataset, collate_fn, Converter
from htr_model.model_v2 import build_model_v2, IAM_ALPHABET


# ────────────────────────────────────────────────────────────────
# Levenshtein alignment  (returns list of edit operations)
# ────────────────────────────────────────────────────────────────
def align(ref, hyp):
    """
    Returns list of (op, ref_char, hyp_char):
      op ∈ {'match', 'sub', 'del', 'ins'}
    Uses standard Levenshtein backtrace.
    """
    n, m = len(ref), len(hyp)
    # DP table
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j],    # deletion
                                   d[i][j-1],    # insertion
                                   d[i-1][j-1])  # substitution

    # Backtrace
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            ops.append(('match', ref[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + 1:
            ops.append(('sub', ref[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i-1][j] + 1:
            ops.append(('del', ref[i-1], None))
            i -= 1
        else:
            ops.append(('ins', None, hyp[j-1]))
            j -= 1
    ops.reverse()
    return ops


def get_predictions(checkpoint_path, device, lmdb_path, batch_size=64, workers=4):
    """Load model and get greedy predictions on dataset."""
    model = build_model_v2(IAM_ALPHABET, hidden_size=512, dropout=0.1)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt))
    model = model.to(device).eval()

    ds     = LMDBDataset(lmdb_path, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=workers, pin_memory=True)
    converter = Converter(IAM_ALPHABET)

    refs, hyps = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            lp     = model(images)
            hyps.extend(converter.decode_batch(lp))
            refs.extend(labels)

    return refs, hyps


def compute_char_stats(refs, hyps):
    """
    Returns dict: char -> {count, sub, del, ins, match}
    'count' = number of times char appears in refs
    """
    stats = defaultdict(lambda: {'count': 0, 'sub': 0, 'del': 0, 'ins': 0, 'match': 0})

    for ref, hyp in zip(refs, hyps):
        for op, rc, hc in align(ref, hyp):
            if op == 'match':
                stats[rc]['count'] += 1
                stats[rc]['match'] += 1
            elif op == 'sub':
                stats[rc]['count'] += 1
                stats[rc]['sub']   += 1
            elif op == 'del':
                stats[rc]['count'] += 1
                stats[rc]['del']   += 1
            elif op == 'ins':
                # insertion: the model produced hc where nothing in ref exists
                stats['[INS]']['ins'] += 1

    return stats


def char_cer(s):
    """CER for a single character = (sub + del + ins) / count"""
    denom = max(s['count'], 1)
    return (s['sub'] + s['del'] + s.get('ins', 0)) / denom * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_a',   required=True, help='Checkpoint A (e.g. exp1_full_iam)')
    parser.add_argument('--ckpt_b',   default=None,  help='Checkpoint B for comparison')
    parser.add_argument('--name_a',   default='model_a')
    parser.add_argument('--name_b',   default='model_b')
    parser.add_argument('--split',    default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--lmdb_dir', default='./data/lmdb')
    parser.add_argument('--gpu',      default='0')
    parser.add_argument('--out_csv',  default='./results/char_errors.csv')
    parser.add_argument('--out_json', default='./results/char_errors.json')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lmdb_path = os.path.join(args.lmdb_dir, args.split)

    print(f'=== Model A: {args.name_a} ===')
    refs_a, hyps_a = get_predictions(args.ckpt_a, device, lmdb_path)
    stats_a = compute_char_stats(refs_a, hyps_a)

    stats_b = None
    if args.ckpt_b:
        print(f'=== Model B: {args.name_b} ===')
        refs_b, hyps_b = get_predictions(args.ckpt_b, device, lmdb_path)
        stats_b = compute_char_stats(refs_b, hyps_b)

    # ── Build output table ──
    all_chars = sorted(set(IAM_ALPHABET) | set(stats_a.keys()))
    rows = []
    for c in all_chars:
        if c == '[INS]':
            continue
        sa = stats_a.get(c, {'count': 0, 'sub': 0, 'del': 0, 'ins': 0, 'match': 0})
        row = {
            'char':         repr(c),
            'char_raw':     c,
            f'{args.name_a}_count':  sa['count'],
            f'{args.name_a}_sub':    sa['sub'],
            f'{args.name_a}_del':    sa['del'],
            f'{args.name_a}_cer':    round(char_cer(sa), 2),
        }
        if stats_b:
            sb = stats_b.get(c, {'count': 0, 'sub': 0, 'del': 0, 'ins': 0, 'match': 0})
            row[f'{args.name_b}_count'] = sb['count']
            row[f'{args.name_b}_sub']   = sb['sub']
            row[f'{args.name_b}_del']   = sb['del']
            row[f'{args.name_b}_cer']   = round(char_cer(sb), 2)
            row['delta_cer'] = round(char_cer(sb) - char_cer(sa), 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(f'{args.name_a}_cer', ascending=False)
    df.to_csv(args.out_csv, index=False)
    print(f'\nSaved → {args.out_csv}')

    # ── Print top 30 ──
    print(f'\n{"Char":6} {"Count":>7} {"A CER%":>8}', end='')
    if stats_b:
        print(f' {"B CER%":>8} {"Δ":>6}', end='')
    print()
    print('-' * (40 if not stats_b else 55))
    for _, row in df.head(30).iterrows():
        c_count = row[f'{args.name_a}_count']
        if c_count < 5:
            continue
        print(f"{row['char']:6} {c_count:>7} {row[f'{args.name_a}_cer']:>8.1f}%", end='')
        if stats_b:
            print(f" {row[f'{args.name_b}_cer']:>8.1f}% {row['delta_cer']:>+6.1f}%", end='')
        print()

    # Overall CER check
    import editdistance
    def overall_cer(refs, hyps):
        e = t = 0
        for r, h in zip(refs, hyps):
            e += editdistance.eval(r, h); t += max(len(r), 1)
        return e / t * 100

    print(f'\nOverall A CER: {overall_cer(refs_a, hyps_a):.4f}%')
    if stats_b:
        print(f'Overall B CER: {overall_cer(refs_b, hyps_b):.4f}%')

    # Save JSON for paper
    result = {
        'model_a': args.name_a,
        'model_b': args.name_b,
        'split': args.split,
        'char_stats': {
            row['char_raw']: {
                f'{args.name_a}_cer': row[f'{args.name_a}_cer'],
                f'{args.name_a}_count': row[f'{args.name_a}_count'],
                **(
                    {f'{args.name_b}_cer': row[f'{args.name_b}_cer'],
                     'delta_cer': row['delta_cer']}
                    if stats_b else {}
                )
            }
            for _, row in df.iterrows()
        }
    }
    with open(args.out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved → {args.out_json}')


if __name__ == '__main__':
    main()
