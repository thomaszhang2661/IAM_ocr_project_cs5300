"""
Split a synthetic LMDB into train/val at a given ratio.

Usage:
    python data/split_synth_lmdb.py \
        --src  data/lmdb/train_synth_full \
        --train data/lmdb/train_synth \
        --val   data/lmdb/val_synth \
        --ratio 0.8 \
        --seed  42
"""

import argparse
import os
import random

import lmdb


def copy_samples(src_env, dst_path, indices):
    os.makedirs(dst_path, exist_ok=True)
    map_size = max(1 << 31, len(indices) * 80 * 1024)
    dst_env = lmdb.open(dst_path, map_size=map_size)

    with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
        for new_i, old_i in enumerate(indices, 1):
            img = src_txn.get(f'image-{old_i:09d}'.encode())
            lbl = src_txn.get(f'label-{old_i:09d}'.encode())
            if img is None or lbl is None:
                continue
            dst_txn.put(f'image-{new_i:09d}'.encode(), img)
            dst_txn.put(f'label-{new_i:09d}'.encode(), lbl)
        dst_txn.put(b'num-samples', str(len(indices)).encode())

    dst_env.close()
    print(f'  Written {len(indices)} samples → {dst_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',   required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--val',   required=True)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--seed',  type=int,   default=42)
    args = parser.parse_args()

    src_env = lmdb.open(args.src, readonly=True, lock=False)
    with src_env.begin() as txn:
        n = int(txn.get(b'num-samples').decode())
    print(f'Source: {n} samples in {args.src}')

    random.seed(args.seed)
    indices = list(range(1, n + 1))
    random.shuffle(indices)

    split = int(len(indices) * args.ratio)
    train_idx = sorted(indices[:split])
    val_idx   = sorted(indices[split:])

    print(f'Split: {len(train_idx)} train / {len(val_idx)} val  (ratio={args.ratio})')
    copy_samples(src_env, args.train, train_idx)
    copy_samples(src_env, args.val,   val_idx)
    src_env.close()
    print('Done.')


if __name__ == '__main__':
    main()
