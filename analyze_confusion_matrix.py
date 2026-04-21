"""
Build model confusion matrix: what does the model predict when it's wrong?

For each (ref_char, pred_char) substitution pair, count occurrences.
Also counts deletions (ref_char → ∅) and insertions (∅ → pred_char).

Usage:
    python analyze_confusion_matrix.py \
        --checkpoint checkpoints/exp3_full_10x/best.pt \
        --run_name exp3_full_10x \
        --out_csv results/confusion_matrix_exp3.csv \
        --top_n 40
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from htr_model.dataset import LMDBDataset, collate_fn, Converter
from htr_model.model_v2 import build_model_v2, IAM_ALPHABET


def align(ref, hyp):
    """Levenshtein backtrace. Returns list of (op, ref_char, hyp_char)."""
    n, m = len(ref), len(hyp)
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            ops.append(('match', ref[i-1], hyp[j-1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i-1][j-1]+1:
            ops.append(('sub', ref[i-1], hyp[j-1])); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i-1][j]+1:
            ops.append(('del', ref[i-1], '∅')); i -= 1
        else:
            ops.append(('ins', '∅', hyp[j-1])); j -= 1
    ops.reverse()
    return ops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--run_name',   default='model')
    parser.add_argument('--split',      default='test')
    parser.add_argument('--lmdb_dir',   default='./data/lmdb')
    parser.add_argument('--gpu',        default='0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--top_n',      type=int, default=40,
                        help='Top N confusion pairs to print')
    parser.add_argument('--min_count',  type=int, default=3)
    parser.add_argument('--out_csv',    default=None)
    parser.add_argument('--out_heatmap_csv', default=None,
                        help='Save full matrix as pivot table for heatmap')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ──
    model = build_model_v2(IAM_ALPHABET, hidden_size=512, dropout=0.1)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt))
    model = model.to(device).eval()
    print(f'Loaded: {args.checkpoint}')

    # ── Dataset ──
    ds     = LMDBDataset(os.path.join(args.lmdb_dir, args.split), augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)
    converter = Converter(IAM_ALPHABET)

    # ── Decode ──
    refs, hyps = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            hyps.extend(converter.decode_batch(model(imgs)))
            refs.extend(labels)

    # ── Build confusion pairs ──
    pair_counts = defaultdict(int)   # (ref_char, pred_char) → count
    ref_char_count = defaultdict(int)

    for ref, hyp in zip(refs, hyps):
        for op, rc, pc in align(ref, hyp):
            if op == 'match':
                ref_char_count[rc] += 1
            elif op == 'sub':
                pair_counts[(rc, pc)] += 1
                ref_char_count[rc]    += 1
            elif op == 'del':
                pair_counts[(rc, '∅')] += 1
                ref_char_count[rc]     += 1
            elif op == 'ins':
                pair_counts[('∅', pc)] += 1

    # ── Build DataFrame ──
    rows = []
    for (rc, pc), cnt in pair_counts.items():
        if cnt < args.min_count:
            continue
        total_rc   = ref_char_count.get(rc, 1)
        error_rate = cnt / total_rc * 100 if rc != '∅' else None
        rows.append({
            'ref_char':   rc,
            'pred_char':  pc,
            'count':      cnt,
            'ref_total':  total_rc,
            'error_rate': round(error_rate, 1) if error_rate is not None else None,
        })

    df = pd.DataFrame(rows).sort_values('count', ascending=False)

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f'Saved pairs → {args.out_csv}')

    # ── Heatmap pivot (substitutions only, top chars) ──
    subs = df[(df['ref_char'] != '∅') & (df['pred_char'] != '∅')].copy()
    if args.out_heatmap_csv:
        pivot = subs.pivot_table(
            index='ref_char', columns='pred_char',
            values='count', fill_value=0
        )
        pivot.to_csv(args.out_heatmap_csv)
        print(f'Saved heatmap → {args.out_heatmap_csv}')

    # ── Print top-N substitution pairs ──
    print(f'\n=== Top {args.top_n} substitution pairs ({args.run_name}) ===')
    print(f'{"Ref":6} → {"Pred":6}  {"Count":>7}  {"Rate%":>8}')
    print('─' * 35)
    subs_top = subs.head(args.top_n)
    for _, row in subs_top.iterrows():
        rc = repr(row['ref_char'])
        pc = repr(row['pred_char'])
        print(f'{rc:6} → {pc:6}  {int(row["count"]):>7}  {row["error_rate"]:>7.1f}%')

    # ── Top deletions ──
    dels = df[df['pred_char'] == '∅'].head(15)
    print(f'\n=== Top deletions (model misses these chars) ===')
    for _, row in dels.iterrows():
        print(f'  {repr(row["ref_char"]):6} deleted  {int(row["count"]):>5}×  ({row["error_rate"]:.1f}% of occurrences)')

    # ── Top insertions ──
    ins = df[df['ref_char'] == '∅'].head(10)
    print(f'\n=== Top insertions (model hallucinates these) ===')
    for _, row in ins.iterrows():
        print(f'  {repr(row["pred_char"]):6} inserted  {int(row["count"]):>5}×')


if __name__ == '__main__':
    main()
