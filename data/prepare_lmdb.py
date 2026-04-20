"""
Convert IAM HuggingFace-format data to LMDB for CRNN training.

Reads from:
    data/iam_hf/{split}/labels.csv   (columns: id, image_path, text)
    data/iam_hf/{split}/images/      (PNG files)

Usage:
    # Prepare all splits (original / full)
    python data/prepare_lmdb.py --hf_root ./data/iam_hf --output_dir ./data/lmdb

    # Also build a clean train split (flagged rows removed)
    python data/prepare_lmdb.py --hf_root ./data/iam_hf --output_dir ./data/lmdb \
        --flagged_csv ./results/doubao_train_flagged.csv
"""

import argparse
import io
import os
import sys

import cv2
import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm

IMG_HEIGHT = 64


def resize_to_h32(img_gray):
    h, w = img_gray.shape
    if h == 0 or w == 0:
        return None
    new_w = max(4, int(w * IMG_HEIGHT / h))
    return cv2.resize(img_gray, (new_w, IMG_HEIGHT), interpolation=cv2.INTER_AREA)


def write_lmdb(samples, output_path, map_size_gb=8):
    """
    samples: list of dict with keys 'image_path' and 'text'
    """
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=map_size_gb * 1024 ** 3)

    count = 0
    skipped = 0
    with env.begin(write=True) as txn:
        for sample in tqdm(samples, desc=f'  {os.path.basename(output_path)}'):
            img = cv2.imread(sample['image_path'], cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                continue
            img = resize_to_h32(img)
            if img is None:
                skipped += 1
                continue

            count += 1
            _, buf = cv2.imencode('.png', img)
            key_img = f'image-{count:09d}'.encode()
            key_lbl = f'label-{count:09d}'.encode()
            txn.put(key_img, buf.tobytes())
            txn.put(key_lbl, sample['text'].encode('utf-8'))

        txn.put(b'num-samples', str(count).encode())

    env.close()
    print(f'  Written: {count}  Skipped: {skipped}')
    return count


def load_csv(csv_path, project_root):
    df = pd.read_csv(csv_path)
    # image_path in CSV may be relative (e.g. ./data/iam_hf/...)
    def resolve(p):
        if os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(project_root, p))
    df['image_path'] = df['image_path'].apply(resolve)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_root',      default='./data/iam_hf',
                        help='Root of HuggingFace IAM download')
    parser.add_argument('--output_dir',   default='./data/lmdb',
                        help='Output directory for LMDB databases')
    parser.add_argument('--flagged_csv',  default=None,
                        help='doubao_train_flagged.csv – rows to exclude from clean split')
    parser.add_argument('--splits',       nargs='+',
                        default=['train', 'validation', 'test'])
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(args.output_dir, exist_ok=True)

    # Load flagged indices for clean split
    flagged_idx = set()
    if args.flagged_csv and os.path.exists(args.flagged_csv):
        df_flag = pd.read_csv(args.flagged_csv)
        flagged_idx = set(df_flag['idx'].tolist())
        print(f'Loaded {len(flagged_idx)} flagged indices from {args.flagged_csv}')

    hf_split_map = {
        'train':      'train',
        'val':        'validation',
        'validation': 'validation',
        'test':       'test',
    }

    for split in args.splits:
        hf_split = hf_split_map.get(split, split)
        csv_path = os.path.join(args.hf_root, hf_split, 'labels.csv')
        if not os.path.exists(csv_path):
            print(f'[SKIP] {csv_path} not found')
            continue

        print(f'\n--- {split} ({hf_split}) ---')
        df = load_csv(csv_path, project_root)
        samples = df[['image_path', 'text']].to_dict('records')

        # Original (full) split
        out_name = 'val' if split == 'validation' else split
        orig_path = os.path.join(args.output_dir, f'{out_name}')
        write_lmdb(samples, orig_path)

        # Clean train split (flagged rows removed)
        if split == 'train' and flagged_idx:
            clean_samples = [s for i, s in enumerate(samples) if i not in flagged_idx]
            removed = len(samples) - len(clean_samples)
            print(f'  Clean split: removed {removed} flagged rows')
            clean_path = os.path.join(args.output_dir, 'train_clean')
            write_lmdb(clean_samples, clean_path)

    print('\nDone.')


if __name__ == '__main__':
    main()
