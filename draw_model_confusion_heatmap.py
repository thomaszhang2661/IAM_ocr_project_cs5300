"""
Draw model prediction confusion matrix heatmap (26×26 lowercase letters)
with numbers annotated on high-count cells.
Generates both Exp3-10x and Exp5-10x versions.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

LETTERS = list('abcdefghijklmnopqrstuvwxyz')
IDX = {c: i for i, c in enumerate(LETTERS)}

def build_matrix(csv_path):
    df = pd.read_csv(csv_path)
    mat = np.zeros((26, 26), dtype=int)
    for _, row in df.iterrows():
        r = str(row['ref_char'])
        p = str(row['pred_char'])
        cnt = int(row['count'])
        if r in IDX and p in IDX and r != p:
            mat[IDX[r], IDX[p]] += cnt
    return mat

def draw_heatmap(mat, title, out_path, annotate_min=30, log_vmax=None):
    """
    Draw heatmap using log1p scale + YlOrRd colormap, matching vlm_confusion_matrix_1to1.png style.
    log_vmax: fixed upper bound for log1p color scale (for consistency across plots).
    """
    fig, ax = plt.subplots(figsize=(13, 11))

    log_mat = np.log1p(mat)
    vmax = log_vmax if log_vmax is not None else log_mat.max()

    im = ax.imshow(log_mat, cmap='YlOrRd', aspect='equal', vmin=0, vmax=vmax)

    # Colorbar with raw-count tick labels
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('log(1+count)', fontsize=10)
    ticks = [t for t in [0, 1, 2, 3, 4, 5, 6] if t <= vmax + 0.01]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(int(np.expm1(t))) for t in ticks])

    # Axes
    ax.set_xticks(range(26))
    ax.set_yticks(range(26))
    ax.set_xticklabels(LETTERS, fontsize=10)
    ax.set_yticklabels(LETTERS, fontsize=10)
    ax.set_xlabel('Predicted character', fontsize=12)
    ax.set_ylabel('Reference character (ground truth)', fontsize=12)
    ax.set_title(title, fontsize=12, pad=10)

    # Annotate high-count cells
    for i in range(26):
        for j in range(26):
            v = mat[i, j]
            if v >= annotate_min:
                text_color = 'white' if log_mat[i, j] / vmax > 0.65 else 'black'
                ax.text(j, i, str(v), ha='center', va='center',
                        fontsize=7, color=text_color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out_path}')

    # Print top pairs
    flat = [(mat[i,j], LETTERS[i], LETTERS[j])
            for i in range(26) for j in range(26) if i != j]
    flat.sort(reverse=True)
    print(f'  Top 15 pairs:')
    for cnt, r, p in flat[:15]:
        print(f"    '{r}'→'{p}': {cnt}")


def main():
    base = os.path.dirname(__file__)
    results = os.path.join(base, 'results')

    configs = [
        ('confusion_pairs_exp3.csv',
         'Model Prediction Confusion Matrix — Exp3-10x (test set)\nRow=ref (ground truth), Col=pred; diagonal excluded',
         'model_confusion_matrix_exp3.png',
         30),
        ('confusion_pairs_exp5.csv',
         'Model Prediction Confusion Matrix — Exp5-10x (test set)\nRow=ref (ground truth), Col=pred; diagonal excluded',
         'model_confusion_matrix_exp5.png',
         30),
    ]

    # Use the same log-scale upper bound for all plots so colors are comparable.
    # VLM matrix max raw count ≈ 99; model max ≈ 280.
    # Shared vmax = log1p(300) ≈ 5.71 gives a consistent scale.
    shared_log_vmax = np.log1p(300)
    print(f'Shared log vmax: {shared_log_vmax:.3f} (= log1p(300))')

    for csv_name, title, out_name, ann_min in configs:
        csv_path = os.path.join(results, csv_name)
        out_path = os.path.join(results, out_name)
        if not os.path.exists(csv_path):
            print(f'Missing: {csv_path}')
            continue
        print(f'\n=== {csv_name} ===')
        mat = build_matrix(csv_path)
        draw_heatmap(mat, title, out_path, annotate_min=ann_min,
                     log_vmax=shared_log_vmax)


if __name__ == '__main__':
    main()
