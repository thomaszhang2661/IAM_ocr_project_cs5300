"""
Download IAM Handwriting Dataset (line level) from HuggingFace.

Source: Teklia/IAM-line  (MIT license)
  - train: 6,480 samples
  - validation: 976 samples
  - test: 2,920 samples

Usage:
    python download_iam.py --output_dir ./data/iam_hf

After running, the data is cached by HuggingFace and also exported as:
    data/iam_hf/
        train/   images + labels.csv
        val/     images + labels.csv
        test/    images + labels.csv
"""

import argparse
import os
import sys

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def export_split(split_data, split_name, output_dir):
    split_dir = os.path.join(output_dir, split_name)
    img_dir   = os.path.join(split_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    records = []
    for idx, sample in enumerate(tqdm(split_data, desc=f'Exporting {split_name}')):
        img_path = os.path.join(img_dir, f'{idx:06d}.png')
        sample['image'].save(img_path)
        records.append({'id': f'{split_name}_{idx:06d}', 'image_path': img_path, 'text': sample['text']})

    df = pd.DataFrame(records)
    csv_path = os.path.join(split_dir, 'labels.csv')
    df.to_csv(csv_path, index=False)
    print(f'  {split_name}: {len(df)} samples → {split_dir}')
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./data/iam_hf',
                        help='Where to save exported images and labels')
    parser.add_argument('--splits', nargs='+', default=['test', 'validation', 'train'],
                        help='Which splits to export (test first for fast iteration)')
    parser.add_argument('--skip_export', action='store_true',
                        help='Download/cache only, skip saving to disk')
    args = parser.parse_args()

    print('Loading IAM-line from HuggingFace (Teklia/IAM-line)...')
    print('(First run downloads ~266 MB, subsequent runs use cache)\n')

    dataset = load_dataset('Teklia/IAM-line')

    print('\nDataset loaded:')
    for split_name, split_data in dataset.items():
        print(f'  {split_name:12s}: {len(split_data):5d} samples')

    if not args.skip_export:
        print(f'\nExporting to {args.output_dir} ...')
        os.makedirs(args.output_dir, exist_ok=True)
        for split_name in args.splits:
            hf_split = 'validation' if split_name == 'val' else split_name
            if hf_split in dataset:
                export_split(dataset[hf_split], split_name, args.output_dir)

    print('\nDone. To verify:')
    print(f'  ls {args.output_dir}')


if __name__ == '__main__':
    main()
