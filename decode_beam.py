"""
Beam search decoding with:
  1. Word-level unigram LM (built from IAM train labels + English word list)
  2. Confusion-matrix smoothing on CTC emissions (optional)

Usage:
    python decode_beam.py --checkpoint checkpoints/exp1_full_iam/best.pt
                          --split test
                          [--beam_width 50]
                          [--alpha 0.5]          # LM weight
                          [--beta 1.0]           # word insertion bonus
                          [--confusion_alpha 0.1] # confusion smoothing weight
                          [--confusion_min 10]   # min count to use pair
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pyctcdecode import build_ctcdecoder

sys.path.insert(0, os.path.dirname(__file__))
from htr_model.dataset import LMDBDataset, collate_fn, Converter
from htr_model.model_v2 import build_model_v2, IAM_ALPHABET

import editdistance


# -----------------------------------------------------------------------
# CER / WER
# -----------------------------------------------------------------------
def compute_cer(refs, hyps):
    errs = total = 0
    for r, h in zip(refs, hyps):
        errs  += editdistance.eval(r, h)
        total += max(len(r), 1)
    return errs / total * 100

def compute_wer(refs, hyps):
    errs = total = 0
    for r, h in zip(refs, hyps):
        rw = r.split(); hw = h.split()
        errs  += editdistance.eval(rw, hw)
        total += max(len(rw), 1)
    return errs / total * 100


# -----------------------------------------------------------------------
# Build word vocabulary from IAM labels
# -----------------------------------------------------------------------
def build_vocabulary(extra_word_lists=True):
    """
    Collect unique words from IAM train labels.
    Optionally supplement with a common English word list.
    """
    words = set()

    # IAM train labels
    try:
        df = pd.read_csv('./data/iam_hf/train/labels.csv')
        for text in df['text'].dropna():
            for w in str(text).split():
                w = w.strip('.,!?;:"\'-()[]').lower()
                if w:
                    words.add(w)
        print(f'  IAM train words: {len(words)}')
    except Exception as e:
        print(f'  Could not load IAM train labels: {e}')

    # IAM val labels
    try:
        df = pd.read_csv('./data/iam_hf/validation/labels.csv')
        for text in df['text'].dropna():
            for w in str(text).split():
                w = w.strip('.,!?;:"\'-()[]').lower()
                if w:
                    words.add(w)
        print(f'  + IAM val words → {len(words)} total')
    except Exception:
        pass

    # Common English words (/usr/share/dict/words or similar)
    if extra_word_lists:
        for wpath in ['/usr/share/dict/words', '/usr/dict/words']:
            if os.path.exists(wpath):
                with open(wpath) as f:
                    for line in f:
                        w = line.strip().lower()
                        if w and w.isalpha():
                            words.add(w)
                print(f'  + system word list → {len(words)} total')
                break

    # Keep only words using IAM alphabet
    valid = set(IAM_ALPHABET)
    filtered = [w for w in words if all(c in valid for c in w) and len(w) >= 1]
    print(f'  Final vocabulary: {len(filtered)} words')
    return filtered


# -----------------------------------------------------------------------
# Confusion matrix emission smoothing
# -----------------------------------------------------------------------
def build_confusion_matrix(confusion_csv, min_count=10):
    """
    Returns a dict {wrong_idx: [(correct_idx, weight), ...]}
    where wrong_idx and correct_idx are positions in IAM_ALPHABET+blank.
    """
    df = pd.read_csv(confusion_csv)
    char2idx = {c: i+1 for i, c in enumerate(IAM_ALPHABET)}  # 0=blank

    confusion = {}  # wrong_idx -> list of (correct_idx, normalized_weight)
    for _, row in df[df['count'] >= min_count].iterrows():
        wc = str(row['wrong_char'])
        cc = str(row['correct_char'])
        if wc not in char2idx or cc not in char2idx:
            continue
        wi = char2idx[wc]
        ci = char2idx[cc]
        if wi not in confusion:
            confusion[wi] = []
        confusion[wi].append((ci, int(row['count'])))

    # Normalize weights per wrong char
    for wi, pairs in confusion.items():
        total = sum(cnt for _, cnt in pairs)
        confusion[wi] = [(ci, cnt / total) for ci, cnt in pairs]

    print(f'  Confusion pairs loaded: {sum(len(v) for v in confusion.values())} '
          f'(min_count={min_count})')
    return confusion


def apply_confusion_smoothing(log_probs_np, confusion, alpha=0.1):
    """
    log_probs_np: (T, C) numpy array of log probabilities
    Adds alpha * P(correct) contribution from confusion pairs.
    Returns smoothed log_probs (T, C).
    """
    if not confusion or alpha == 0:
        return log_probs_np

    T, C = log_probs_np.shape
    probs = np.exp(log_probs_np)  # (T, C)
    smoothed = probs.copy()

    for wrong_idx, pairs in confusion.items():
        if wrong_idx >= C:
            continue
        for correct_idx, weight in pairs:
            if correct_idx >= C:
                continue
            # Transfer alpha * weight * P(wrong) to P(correct)
            delta = alpha * weight * probs[:, wrong_idx]
            smoothed[:, correct_idx] += delta
            smoothed[:, wrong_idx]   -= delta

    smoothed = np.clip(smoothed, 1e-30, None)
    # Renormalize
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    return np.log(smoothed)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',      required=True)
    parser.add_argument('--split',           default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--lmdb_dir',        default='./data/lmdb')
    parser.add_argument('--beam_width',      type=int,   default=50)
    parser.add_argument('--alpha',           type=float, default=0.5,
                        help='LM weight (word unigram)')
    parser.add_argument('--beta',            type=float, default=1.0,
                        help='Word insertion bonus')
    parser.add_argument('--confusion_alpha', type=float, default=0.1,
                        help='Confusion matrix smoothing weight (0=disabled)')
    parser.add_argument('--confusion_min',   type=int,   default=10,
                        help='Min confusion count to apply smoothing')
    parser.add_argument('--confusion_csv',   default='./results/vlm_char_confusion.csv')
    parser.add_argument('--gpu',             default='0')
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--workers',         type=int,   default=4)
    parser.add_argument('--out_json',        default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- Model ----
    model = build_model_v2(IAM_ALPHABET, hidden_size=512, dropout=0.1)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Support both raw state_dict and full checkpoint
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f'Loaded checkpoint: {args.checkpoint}')

    # ---- Dataset ----
    lmdb_path = os.path.join(args.lmdb_dir, args.split)
    ds     = LMDBDataset(lmdb_path, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.workers,
                        pin_memory=True)
    print(f'Dataset: {args.split}  ({len(ds)} samples)')

    # ---- Build vocabulary & decoder ----
    print('\nBuilding vocabulary...')
    vocab = build_vocabulary()

    # pyctcdecode expects labels = [blank] + alphabet
    labels = [''] + list(IAM_ALPHABET)   # index 0 = blank

    decoder = build_ctcdecoder(
        labels     = labels,
        unigrams   = vocab,
        alpha      = args.alpha,
        beta       = args.beta,
    )
    print(f'Decoder built  beam_width={args.beam_width}  '
          f'alpha={args.alpha}  beta={args.beta}')

    # ---- Confusion smoothing ----
    confusion = {}
    if args.confusion_alpha > 0 and os.path.exists(args.confusion_csv):
        print('\nLoading confusion matrix...')
        confusion = build_confusion_matrix(args.confusion_csv, args.confusion_min)

    # ---- Greedy + Beam search in one pass ----
    print('\nRunning greedy + beam search...')
    converter = Converter(IAM_ALPHABET)
    greedy_refs, greedy_hyps, beam_hyps = [], [], []
    sample_idx = 0

    with torch.no_grad():
        for images, labels_batch in loader:
            images    = images.to(device)
            log_probs = model(images)           # (T, N, C)

            # Greedy
            greedy_hyps.extend(converter.decode_batch(log_probs))
            greedy_refs.extend(labels_batch)

            # Beam search per sample (T varies per batch so process here)
            lp_np = log_probs.cpu().numpy()     # (T, N, C)
            T, N, C = lp_np.shape
            for i in range(N):
                seq = lp_np[:, i, :]            # (T, C)
                if confusion:
                    seq = apply_confusion_smoothing(seq, confusion, args.confusion_alpha)
                text = decoder.decode(seq, beam_width=args.beam_width)
                beam_hyps.append(text)

            sample_idx += N
            if sample_idx % 200 == 0:
                print(f'  {sample_idx}/{len(ds)}')

    greedy_cer = compute_cer(greedy_refs, greedy_hyps)
    greedy_wer = compute_wer(greedy_refs, greedy_hyps)
    print(f'Greedy  CER={greedy_cer:.2f}%  WER={greedy_wer:.2f}%')

    print(f'\nBeam search complete.')
    beam_cer = compute_cer(greedy_refs, beam_hyps)
    beam_wer = compute_wer(greedy_refs, beam_hyps)
    print(f'\nBeam    CER={beam_cer:.2f}%  WER={beam_wer:.2f}%')
    print(f'Improvement: CER {greedy_cer - beam_cer:+.2f}%  '
          f'WER {greedy_wer - beam_wer:+.2f}%')

    # ---- Save results ----
    out = {
        'checkpoint':       args.checkpoint,
        'split':            args.split,
        'n_samples':        len(ds),
        'beam_width':       args.beam_width,
        'alpha':            args.alpha,
        'beta':             args.beta,
        'confusion_alpha':  args.confusion_alpha,
        'confusion_min':    args.confusion_min,
        'greedy_cer':       round(greedy_cer, 4),
        'greedy_wer':       round(greedy_wer, 4),
        'beam_cer':         round(beam_cer,   4),
        'beam_wer':         round(beam_wer,   4),
    }
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Results saved → {args.out_json}')

    # Show a few examples
    print('\n--- Sample comparisons ---')
    for i in range(min(5, len(greedy_refs))):
        r = greedy_refs[i]
        g = greedy_hyps[i]
        b = beam_hyps[i]
        g_cer = editdistance.eval(r, g) / max(len(r), 1) * 100
        b_cer = editdistance.eval(r, b) / max(len(r), 1) * 100
        print(f'  REF:   {r}')
        print(f'  Greedy ({g_cer:.0f}%): {g}')
        print(f'  Beam   ({b_cer:.0f}%): {b}')
        print()


if __name__ == '__main__':
    main()
