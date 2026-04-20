"""
Extended confusion analysis: captures 1:1, n:1, 1:n, and n:m confusions.

Outputs:
  results/vlm_confusion_all.csv      — all confusion types
  results/vlm_confusion_multi.csv    — only multi-char confusions (n≠1 or m≠1)
  results/vlm_confusion_matrix_v2.png — updated 26×26 heatmap (1:1 only, for viz)
"""

import difflib
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPLITS = {
    'train': './results/doubao_train_flagged.csv',
    'val':   './results/doubao_val_flagged.csv',
    'test':  './results/doubao_test_flagged.csv',
}

IAM_ALPHABET = (
    ' !"#&\'()*+,-./0123456789:;?'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abcdefghijklmnopqrstuvwxyz'
)
VALID = set(IAM_ALPHABET)


def all_ops(a, b):
    """
    Return all edit operations between strings a (GT) and b (corrected).
    Each op is (wrong_seq, correct_seq) where either or both can be multi-char.
    Includes:
      replace (1:1, n:1, 1:n, n:m)
      delete  (n:0)  — chars in GT that shouldn't be there
      insert  (0:n)  — chars missing from GT
    """
    ops = []
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        wrong   = a[i1:i2]   # what IAM wrote (may be empty for insert)
        correct = b[j1:j2]   # what it should be (may be empty for delete)
        ops.append((wrong, correct))
    return ops


def main():
    os.makedirs('./results', exist_ok=True)

    all_ops_counter  = Counter()   # (wrong_seq, correct_seq) → count
    sub1_counter     = Counter()   # 1:1 only
    multi_counter    = Counter()   # anything else

    for split, csv_path in SPLITS.items():
        df = pd.read_csv(csv_path)
        print(f'{split}: {len(df)} flagged')
        for _, row in df.iterrows():
            gt  = str(row['ground_truth'])  if pd.notna(row['ground_truth'])  else ''
            cor = str(row['corrected_text']) if pd.notna(row['corrected_text']) else ''
            if not gt or not cor:
                continue
            for wrong, correct in all_ops(gt, cor):
                # Filter: only ops involving valid IAM chars
                if not all(c in VALID for c in wrong + correct):
                    continue
                if wrong == correct:
                    continue
                all_ops_counter[(wrong, correct)] += 1
                if len(wrong) == 1 and len(correct) == 1:
                    sub1_counter[(wrong, correct)] += 1
                else:
                    multi_counter[(wrong, correct)] += 1

    # -----------------------------------------------------------------------
    # Save full CSV
    # -----------------------------------------------------------------------
    rows = []
    for (wrong, correct), count in all_ops_counter.most_common():
        op_type = (
            'substitute_1:1' if len(wrong)==1 and len(correct)==1 else
            'merge_n:1'      if len(wrong)>1  and len(correct)==1 else
            'split_1:n'      if len(wrong)==1  and len(correct)>1  else
            'delete_n:0'     if len(correct)==0 else
            'insert_0:n'     if len(wrong)==0   else
            f'replace_{len(wrong)}:{len(correct)}'
        )
        rows.append({
            'wrong':    wrong,
            'correct':  correct,
            'count':    count,
            'type':     op_type,
            'wrong_repr':   repr(wrong),
            'correct_repr': repr(correct),
        })

    df_all = pd.DataFrame(rows)
    df_all.to_csv('./results/vlm_confusion_all.csv', index=False)
    print(f'\nTotal ops: {len(df_all)}  (total count: {df_all["count"].sum()})')

    # -----------------------------------------------------------------------
    # Multi-char only
    # -----------------------------------------------------------------------
    df_multi = df_all[df_all['type'] != 'substitute_1:1'].copy()
    df_multi = df_multi.sort_values('count', ascending=False)
    df_multi.to_csv('./results/vlm_confusion_multi.csv', index=False)

    print(f'\nMulti-char ops: {len(df_multi)}  (count: {df_multi["count"].sum()})')
    print('\nTop 30 multi-char confusions:')
    print(df_multi.head(30)[['wrong','correct','count','type']].to_string(index=False))

    # -----------------------------------------------------------------------
    # Type breakdown
    # -----------------------------------------------------------------------
    print('\nBreakdown by type:')
    for t, g in df_all.groupby('type'):
        print(f'  {t:25s}: {len(g):5d} pairs, {g["count"].sum():6d} total events')

    # -----------------------------------------------------------------------
    # 26×26 heatmap (1:1 lowercase only) — same as before but now complete
    # -----------------------------------------------------------------------
    letters = list('abcdefghijklmnopqrstuvwxyz')
    mat = np.zeros((26, 26), dtype=int)
    for (wrong, correct), count in sub1_counter.items():
        if wrong in letters and correct in letters:
            i = ord(wrong)   - ord('a')
            j = ord(correct) - ord('a')
            mat[i, j] = count

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(np.log1p(mat), cmap='YlOrRd', aspect='equal')
    ax.set_xticks(range(26)); ax.set_yticks(range(26))
    ax.set_xticklabels(letters, fontsize=10)
    ax.set_yticklabels(letters, fontsize=10)
    ax.set_xlabel('Correct character (VLM correction)', fontsize=12)
    ax.set_ylabel('Wrong character (IAM ground truth)', fontsize=12)
    ax.set_title(
        'IAM Annotation Confusion Matrix — lowercase 1:1 substitutions\n'
        '(Multi-char confusions excluded; see vlm_confusion_multi.csv)',
        fontsize=12)
    for i in range(26):
        for j in range(26):
            if mat[i, j] > 10:
                ax.text(j, i, str(mat[i, j]), ha='center', va='center',
                        fontsize=7, color='black' if mat[i,j]<80 else 'white')
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('log(1+count)', fontsize=10)
    ticks = [0,1,2,3,4,5]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(int(np.expm1(t))) for t in ticks])
    plt.tight_layout()
    plt.savefig('./results/vlm_confusion_matrix_v2.png', dpi=150, bbox_inches='tight')
    print('\nSaved heatmap → results/vlm_confusion_matrix_v2.png')


if __name__ == '__main__':
    main()
