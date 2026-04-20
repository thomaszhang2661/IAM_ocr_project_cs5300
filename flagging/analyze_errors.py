"""
Analyze flagged samples after manual review.

Expects flagged_for_review.csv with an 'error_type' column filled in.

Usage:
    python analyze_errors.py \
        --flagged_csv results/flagging/flagged_for_review.csv \
        --all_csv     results/flagging/all_results.csv \
        --output_dir  results/analysis
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from flagging.flag_samples import ERROR_TYPES


def plot_error_distribution(df_flagged, output_dir):
    """Bar chart of error type distribution."""
    counts = df_flagged['error_type'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title('Flagged Sample Error Type Distribution', fontsize=14)
    ax.set_xlabel('Error Type')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    path = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_cer_histogram(df_all, output_dir):
    """CER distribution for all samples."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in [
        (axes[0], 'cer_claude', 'Claude CER Distribution'),
        (axes[1], 'cer_gpt4o',  'GPT-4o CER Distribution'),
    ]:
        ax.hist(df_all[col], bins=50, color='coral', edgecolor='black', alpha=0.8)
        ax.axvline(df_all[col].median(), color='navy', linestyle='--',
                   label=f'median={df_all[col].median():.3f}')
        ax.set_title(title)
        ax.set_xlabel('CER')
        ax.set_ylabel('Count')
        ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'cer_histogram.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_writer_analysis(df_all, df_flagged, output_dir, top_n=20):
    """
    Per-writer flagged rate.
    Useful for identifying writers with systematically bad annotations.
    """
    total_per_writer  = df_all.groupby('writer').size().rename('total')
    flag_per_writer   = df_flagged.groupby('writer').size().rename('flagged')
    writer_df = pd.concat([total_per_writer, flag_per_writer], axis=1).fillna(0)
    writer_df['flag_rate'] = writer_df['flagged'] / writer_df['total']
    writer_df = writer_df.sort_values('flag_rate', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(writer_df.index, writer_df['flag_rate'], color='teal', edgecolor='black')
    ax.set_title(f'Top {top_n} Writers by Flagging Rate', fontsize=14)
    ax.set_xlabel('Writer ID')
    ax.set_ylabel('Flag Rate')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    path = os.path.join(output_dir, 'writer_flagging_rate.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return writer_df


def generate_report(df_all, df_flagged, output_dir):
    """Generate a text/JSON summary report."""
    total = len(df_all)
    flagged = len(df_flagged)

    error_counts = (
        df_flagged['error_type'].value_counts().to_dict()
        if 'error_type' in df_flagged.columns and not df_flagged.empty
        else {}
    )

    report = {
        'total_samples': total,
        'flagged': flagged,
        'flagged_rate': round(flagged / max(total, 1), 4),
        'error_type_counts': error_counts,
        'error_type_rates': {
            k: round(v / max(flagged, 1), 4)
            for k, v in error_counts.items()
        },
        'corpus_cer_claude': round(df_all['cer_claude'].mean(), 4),
        'corpus_cer_gpt4o':  round(df_all['cer_gpt4o'].mean(), 4),
        'median_cer_claude':  round(df_all['cer_claude'].median(), 4),
        'median_cer_gpt4o':   round(df_all['cer_gpt4o'].median(), 4),
        'unique_flagged_writers': df_flagged['writer'].nunique() if not df_flagged.empty else 0,
        'unique_total_writers':   df_all['writer'].nunique(),
    }

    path = os.path.join(output_dir, 'analysis_report.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {path}")

    print("\n=== ANALYSIS REPORT ===")
    print(f"Total samples:    {total}")
    print(f"Flagged:          {flagged} ({report['flagged_rate']*100:.1f}%)")
    print(f"Mean CER Claude:  {report['corpus_cer_claude']:.4f}")
    print(f"Mean CER GPT-4o:  {report['corpus_cer_gpt4o']:.4f}")
    if error_counts:
        print("Error breakdown:")
        for etype, cnt in sorted(error_counts.items(), key=lambda x: -x[1]):
            pct = cnt / max(flagged, 1) * 100
            print(f"  {etype:20s}: {cnt:4d} ({pct:.1f}%)")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flagged_csv', required=True)
    parser.add_argument('--all_csv', required=True)
    parser.add_argument('--output_dir', default='results/analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df_all     = pd.read_csv(args.all_csv)
    df_flagged = pd.read_csv(args.flagged_csv)

    # Filter to only manually reviewed (error_type filled in)
    if 'error_type' in df_flagged.columns:
        df_reviewed = df_flagged[df_flagged['error_type'].notna() &
                                  (df_flagged['error_type'] != '')]
        print(f"Manually reviewed: {len(df_reviewed)} / {len(df_flagged)} flagged")
    else:
        df_reviewed = df_flagged

    plot_cer_histogram(df_all, args.output_dir)
    if not df_reviewed.empty:
        plot_error_distribution(df_reviewed, args.output_dir)
    plot_writer_analysis(df_all, df_flagged, args.output_dir)
    generate_report(df_all, df_reviewed, args.output_dir)


if __name__ == '__main__':
    main()
