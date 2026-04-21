"""
Beam search decoding v2:
  1. Word 4-gram KenLM (trained from public corpus)
  2. Multi-character confusion post-correction (vlm_confusion_multi.csv)
     - applies high-confidence substitutions that improve char n-gram LM score

Usage:
    python decode_beam_v2.py --checkpoint checkpoints/exp3_full_10x/best.pt \\
                             --split test \\
                             --lm_path lm/word_4gram.binary \\
                             --confusion_csv results/vlm_confusion_multi.csv \\
                             --out_json results/beam_v2_exp3_full_10x.json
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import editdistance
import numpy as np
import torch
from torch.utils.data import DataLoader
from pyctcdecode import build_ctcdecoder

sys.path.insert(0, os.path.dirname(__file__))
from htr_model.dataset import LMDBDataset, collate_fn, Converter
from htr_model.model_v2 import build_model_v2, IAM_ALPHABET


# ───────────────────────────────────────────────────────────────────────────────
# CER / WER
# ───────────────────────────────────────────────────────────────────────────────
def compute_cer(refs, hyps):
    errs = total = 0
    for r, h in zip(refs, hyps):
        errs  += editdistance.eval(r, h)
        total += max(len(r), 1)
    return errs / total * 100

def compute_wer(refs, hyps):
    errs = total = 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        errs  += editdistance.eval(rw, hw)
        total += max(len(rw), 1)
    return errs / total * 100


# ───────────────────────────────────────────────────────────────────────────────
# Multi-character confusion post-correction
# ───────────────────────────────────────────────────────────────────────────────
def load_confusion_rules(csv_path, min_count=10, max_len=4):
    """
    Load multi-character confusion rules.
    Returns list of (wrong, correct, count) sorted by count desc.
    Filters:
      - min_count: minimum occurrence
      - max_len: max length of wrong/correct strings (avoid very long rules)
      - Skip pure insertion/deletion unless short
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    rules = []
    for _, row in df.iterrows():
        wrong   = str(row['wrong'])   if not _is_nan(row['wrong'])   else ''
        correct = str(row['correct']) if not _is_nan(row['correct']) else ''
        count   = int(row['count'])
        rtype   = str(row.get('type', ''))

        if count < min_count:
            continue
        # Skip rules where wrong is empty (pure insertion) — can't match in string
        if not wrong:
            continue
        # Skip very long patterns (unreliable)
        if len(wrong) > max_len or len(correct) > max_len:
            continue
        # Skip if wrong == correct
        if wrong == correct:
            continue

        rules.append((wrong, correct, count))

    # Sort by count desc, then by len(wrong) desc (longer patterns first to avoid partial overlap)
    rules.sort(key=lambda x: (-x[2], -len(x[0])))
    print(f'  Loaded {len(rules)} confusion rules (min_count={min_count})')
    return rules


def _is_nan(v):
    try:
        import math
        return math.isnan(float(v))
    except Exception:
        return False


def build_char_ngram_scorer(corpus_lines, order=5):
    """
    Build a simple character n-gram scorer from corpus lines.
    Used to decide whether applying a confusion rule improves the hypothesis.
    Returns a scorer function: score(text) -> float (higher is better)
    """
    from collections import Counter
    import math

    char_counts = [Counter() for _ in range(order + 1)]
    for line in corpus_lines:
        chars = list(line)
        for n in range(1, order + 1):
            for i in range(len(chars) - n + 1):
                gram = tuple(chars[i:i+n])
                char_counts[n][gram] += 1

    total_chars = sum(char_counts[1].values())
    vocab_size  = len(char_counts[1])

    def score(text, n=order):
        """Return average log-prob per character (stupid backoff)."""
        chars  = list(text)
        log_p  = 0.0
        count  = 0
        for i in range(len(chars)):
            best_log_p = None
            for k in range(min(n, i + 1), 0, -1):
                gram = tuple(chars[i - k + 1:i + 1])
                ctx  = gram[:-1]
                ctx_count = char_counts[k - 1][ctx] if k > 1 else total_chars
                c = char_counts[k].get(gram, 0)
                if c > 0 or k == 1:
                    # Laplace smoothed
                    p = (c + 1) / (ctx_count + vocab_size)
                    best_log_p = math.log(p)
                    break
                # backoff with factor 0.4 (stupid backoff)
            if best_log_p is None:
                best_log_p = math.log(1 / vocab_size)
            log_p += best_log_p
            count += 1
        return log_p / max(count, 1)

    return score


def apply_confusion_corrections(text, rules, scorer, min_improvement=0.0005):
    """
    Greedily apply confusion rules that improve the char n-gram score.
    Processes each position left-to-right; accepts first improving rule found.
    """
    result = text
    base_score = scorer(result)

    i = 0
    while i < len(result):
        applied = False
        for wrong, correct, count in rules:
            wl = len(wrong)
            if result[i:i+wl] == wrong:
                candidate = result[:i] + correct + result[i+wl:]
                new_score = scorer(candidate)
                if new_score > base_score + min_improvement:
                    result     = candidate
                    base_score = new_score
                    applied    = True
                    # Don't advance i — re-check same position with new text
                    break
        if not applied:
            i += 1

    return result


# ───────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ───────────────────────────────────────────────────────────────────────────────
def build_vocabulary():
    """
    Build word vocabulary preserving original case from IAM labels.
    IAM contains many proper nouns (London, Mr, Manchester) that only
    appear capitalized — lowercasing them causes unk_score_offset penalties
    during beam search.
    Strategy: add each word in its original case AND in lowercase.
    """
    import pandas as pd
    words = set()
    STRIP = str.maketrans('', '', '.,!?;:"\'-()[]#&*+/')

    # IAM train + val labels — keep original case for proper nouns
    for csv_path in ['./data/iam_hf/train/labels.csv',
                     './data/iam_hf/validation/labels.csv']:
        try:
            df = pd.read_csv(csv_path)
            for text in df['text'].dropna():
                for w in str(text).split():
                    w_stripped = w.translate(STRIP)
                    if w_stripped:
                        words.add(w_stripped)           # original case
                        words.add(w_stripped.lower())   # lowercase fallback
        except Exception:
            pass

    iam_count = len(words)
    print(f'  IAM vocab: {iam_count} entries')

    # LLM-generated rare-character text (if available)
    for extra in ['./data/rare_texts.txt', './lm/corpus_iam.txt']:
        if os.path.exists(extra):
            with open(extra) as f:
                for line in f:
                    for w in line.split():
                        w_stripped = w.translate(STRIP)
                        if w_stripped:
                            words.add(w_stripped)
                            words.add(w_stripped.lower())

    # System dictionary (lowercase common English words)
    for wpath in ['/usr/share/dict/words', '/usr/dict/words']:
        if os.path.exists(wpath):
            with open(wpath) as f:
                for line in f:
                    w = line.strip()
                    if w and w.isalpha():
                        words.add(w)
                        words.add(w.lower())
            break

    valid = set(IAM_ALPHABET)
    filtered = [w for w in words if len(w) >= 1 and all(c in valid for c in w)]
    print(f'  Total vocabulary: {len(filtered)} entries '
          f'(+{len(filtered)-iam_count} from dict/extra)')
    return filtered


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',      required=True)
    parser.add_argument('--split',           default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--lmdb_dir',        default='./data/lmdb')
    parser.add_argument('--lm_path',         default='./lm/word_4gram.binary',
                        help='KenLM binary or ARPA path')
    parser.add_argument('--confusion_csv',   default='./results/vlm_confusion_multi.csv')
    parser.add_argument('--confusion_min',   type=int,   default=15,
                        help='Min count for confusion rules')
    parser.add_argument('--beam_width',      type=int,   default=50)
    parser.add_argument('--alpha',           type=float, default=0.5)
    parser.add_argument('--beta',            type=float, default=1.0)
    parser.add_argument('--gpu',             default='0')
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--workers',         type=int,   default=4)
    parser.add_argument('--out_json',        default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model_v2(IAM_ALPHABET, hidden_size=512, dropout=0.1)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt))
    model = model.to(device).eval()
    print(f'Loaded: {args.checkpoint}')

    # ── Dataset ────────────────────────────────────────────────────────────────
    lmdb_path = os.path.join(args.lmdb_dir, args.split)
    ds     = LMDBDataset(lmdb_path, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.workers,
                        pin_memory=True)
    print(f'Dataset: {args.split} ({len(ds)} samples)')

    # ── Build decoder ──────────────────────────────────────────────────────────
    print('\nBuilding vocabulary...')
    vocab = build_vocabulary()

    labels = [''] + list(IAM_ALPHABET)   # index 0 = blank

    if os.path.exists(args.lm_path):
        print(f'Using KenLM: {args.lm_path}')
        decoder = build_ctcdecoder(
            labels,
            kenlm_model_path = args.lm_path,
            unigrams         = vocab,
            alpha            = args.alpha,
            beta             = args.beta,
        )
    else:
        print(f'WARNING: LM not found at {args.lm_path}, using unigram only')
        decoder = build_ctcdecoder(labels, unigrams=vocab,
                                   alpha=args.alpha, beta=args.beta)

    # ── Confusion rules + char n-gram scorer ───────────────────────────────────
    confusion_rules = []
    char_scorer     = None
    if os.path.exists(args.confusion_csv):
        print('\nLoading confusion rules...')
        confusion_rules = load_confusion_rules(args.confusion_csv,
                                               min_count=args.confusion_min)
        print('Building char n-gram scorer from IAM corpus...')
        iam_lines = []
        for csv_path in ['./data/iam_hf/train/labels.csv',
                         './data/iam_hf/validation/labels.csv']:
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                iam_lines.extend(df['text'].dropna().tolist())
            except Exception:
                pass
        # Supplement with public corpus if available
        pub = './lm/corpus_public.txt'
        if os.path.exists(pub):
            with open(pub) as f:
                iam_lines.extend(f.read().splitlines()[:50000])
        char_scorer = build_char_ngram_scorer(iam_lines, order=5)
        print(f'  Char scorer built on {len(iam_lines)} lines')

    # ── Greedy + Beam + Post-correction ───────────────────────────────────────
    print('\nDecoding...')
    converter = Converter(IAM_ALPHABET)
    refs, greedy_hyps, beam_hyps, corrected_hyps = [], [], [], []

    with torch.no_grad():
        done = 0
        for images, labels_batch in loader:
            images    = images.to(device)
            log_probs = model(images)   # (T, N, C)

            greedy_hyps.extend(converter.decode_batch(log_probs))
            refs.extend(labels_batch)

            lp_np = log_probs.cpu().numpy()   # (T, N, C)
            T, N, C = lp_np.shape
            for i in range(N):
                seq  = lp_np[:, i, :]
                text = decoder.decode(seq, beam_width=args.beam_width)
                beam_hyps.append(text)
                if confusion_rules and char_scorer:
                    corrected = apply_confusion_corrections(
                        text, confusion_rules, char_scorer)
                    corrected_hyps.append(corrected)
                else:
                    corrected_hyps.append(text)

            done += N
            if done % 500 == 0:
                print(f'  {done}/{len(ds)}')

    greedy_cer = compute_cer(refs, greedy_hyps)
    greedy_wer = compute_wer(refs, greedy_hyps)
    beam_cer   = compute_cer(refs, beam_hyps)
    beam_wer   = compute_wer(refs, beam_hyps)
    corr_cer   = compute_cer(refs, corrected_hyps)
    corr_wer   = compute_wer(refs, corrected_hyps)

    print(f'\n{"="*50}')
    print(f'Greedy   CER={greedy_cer:.2f}%  WER={greedy_wer:.2f}%')
    print(f'Beam     CER={beam_cer:.2f}%  WER={beam_wer:.2f}%  '
          f'(Δ={beam_cer-greedy_cer:+.2f}%)')
    print(f'Corrected CER={corr_cer:.2f}%  WER={corr_wer:.2f}%  '
          f'(Δ vs beam={corr_cer-beam_cer:+.2f}%)')

    out = {
        'checkpoint':   args.checkpoint,
        'split':        args.split,
        'n_samples':    len(ds),
        'lm_path':      args.lm_path,
        'beam_width':   args.beam_width,
        'alpha':        args.alpha,
        'beta':         args.beta,
        'confusion_min': args.confusion_min,
        'greedy_cer':   round(greedy_cer, 4),
        'greedy_wer':   round(greedy_wer, 4),
        'beam_cer':     round(beam_cer,   4),
        'beam_wer':     round(beam_wer,   4),
        'corrected_cer': round(corr_cer,  4),
        'corrected_wer': round(corr_wer,  4),
    }
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Saved → {args.out_json}')

    # ── Sample comparisons ────────────────────────────────────────────────────
    print('\n--- Samples ---')
    for i in range(min(8, len(refs))):
        r = refs[i]; g = greedy_hyps[i]; b = beam_hyps[i]; c = corrected_hyps[i]
        if g != b or b != c:   # only show changed ones
            print(f'  REF:  {r}')
            if g != b:
                print(f'  GRD:  {g}')
            print(f'  BEAM: {b}')
            if b != c:
                print(f'  CORR: {c}')
            print()


if __name__ == '__main__':
    main()
