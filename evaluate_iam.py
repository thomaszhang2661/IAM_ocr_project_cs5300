"""
Evaluate a trained CRNN checkpoint on the IAM test split.

Usage:
  python evaluate_iam.py --checkpoint checkpoints/original/best.pt --run_name original
  python evaluate_iam.py --checkpoint checkpoints/clean/best.pt    --run_name clean
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from htr_model.dataset import LMDBDataset, collate_fn, Converter


def edit_distance(a, b):
    import editdistance
    return editdistance.eval(a, b)


def compute_cer_wer(refs, hyps):
    char_errors = char_total = word_errors = word_total = 0
    for ref, hyp in zip(refs, hyps):
        char_errors += edit_distance(ref, hyp)
        char_total  += max(len(ref), 1)
        ref_w = ref.split()
        hyp_w = hyp.split()
        word_errors += edit_distance(ref_w, hyp_w)
        word_total  += max(len(ref_w), 1)
    cer = char_errors / char_total * 100
    wer = word_errors / word_total * 100
    return cer, wer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--run_name',   default='eval')
    parser.add_argument('--lmdb_dir',   default='./data/lmdb')
    parser.add_argument('--split',      default='test',
                        help='LMDB split name (test or test_clean)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model',      choices=['v1', 'v2'], default='v1')
    parser.add_argument('--hidden',     type=int, default=256)
    parser.add_argument('--workers',    type=int, default=4)
    parser.add_argument('--gpu',        default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    test_lmdb = os.path.join(args.lmdb_dir, args.split)
    test_ds   = LMDBDataset(test_lmdb)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn,
                             num_workers=args.workers, pin_memory=True)
    print(f'Test samples: {len(test_ds)}')

    if args.model == 'v2':
        from htr_model.model_v2 import build_model_v2, IAM_ALPHABET
        model = build_model_v2(IAM_ALPHABET, hidden_size=args.hidden).to(device)
    else:
        from htr_model.model import build_model, IAM_ALPHABET
        model = build_model(IAM_ALPHABET, hidden_size=args.hidden).to(device)
    converter = Converter(IAM_ALPHABET)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f'Loaded: {args.checkpoint}')

    refs, hyps = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            log_probs = model(images)           # (T, N, C)
            decoded = converter.decode_batch(log_probs)
            refs.extend(labels)
            hyps.extend(decoded)

    cer, wer = compute_cer_wer(refs, hyps)

    print(f'\n=== Test Results: {args.run_name} ===')
    print(f'  Samples : {len(refs)}')
    print(f'  CER     : {cer:.2f}%')
    print(f'  WER     : {wer:.2f}%')

    os.makedirs('results', exist_ok=True)
    out = {
        'run_name':     args.run_name,
        'checkpoint':   args.checkpoint,
        'test_samples': len(refs),
        'cer_pct':      round(cer, 4),
        'wer_pct':      round(wer, 4),
    }
    out_path = f'results/test_results_{args.run_name}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  Saved   : {out_path}')


if __name__ == '__main__':
    main()
