"""
Build val_clean LMDB by removing VLM-flagged validation samples.
Usage: python data/create_val_clean_lmdb.py
"""
import os
import sys
import pandas as pd
import lmdb

VAL_LMDB       = './data/lmdb/val'
VAL_CLEAN_LMDB = './data/lmdb/val_clean'
FLAGGED_CSV    = './results/doubao_val_flagged.csv'


def main():
    # Load flagged indices
    df_flag = pd.read_csv(FLAGGED_CSV)
    flagged_idx = set(df_flag['idx'].tolist())
    print(f'Flagged val indices: {len(flagged_idx)}')

    # Open source LMDB
    src_env = lmdb.open(VAL_LMDB, readonly=True, lock=False)
    with src_env.begin() as txn:
        n_total = int(txn.get(b'num-samples'))
    print(f'Source val LMDB: {n_total} samples')

    # Write clean LMDB
    os.makedirs(VAL_CLEAN_LMDB, exist_ok=True)
    dst_env = lmdb.open(VAL_CLEAN_LMDB, map_size=1 << 30)  # 1 GB

    kept = 0
    with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
        for i in range(1, n_total + 1):
            orig_idx = i - 1  # 0-based index in original dataset
            if orig_idx in flagged_idx:
                continue
            img = src_txn.get(f'image-{i:09d}'.encode())
            lbl = src_txn.get(f'label-{i:09d}'.encode())
            if img is None or lbl is None:
                continue
            kept += 1
            dst_txn.put(f'image-{kept:09d}'.encode(), img)
            dst_txn.put(f'label-{kept:09d}'.encode(), lbl)
        dst_txn.put(b'num-samples', str(kept).encode())

    src_env.close()
    dst_env.close()
    print(f'val_clean LMDB: {kept} samples (removed {n_total - kept} flagged)')
    print(f'Saved → {VAL_CLEAN_LMDB}')


if __name__ == '__main__':
    main()
