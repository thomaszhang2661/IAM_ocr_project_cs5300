"""
Evaluate beam + word-level confusion spell correction.

Pipeline:
  1. Run CRNN-CTC model (greedy + beam)
  2. Apply ConfusionSpellCorrector (word-level, vocab-gated)
  3. Report CER at each stage

Usage:
  python eval_spell_corrector.py \
    --checkpoint checkpoints/exp3_full_10x/best.pt \
    --confusion_csv results/model_confusion_exp3.csv \
    --confusion_min 5 \
    --run_name exp3_full_10x \
    --gpu 0
"""

import argparse, json, os, sys
import editdistance
import torch
from torch.utils.data import DataLoader
from pyctcdecode import build_ctcdecoder

sys.path.insert(0, os.path.dirname(__file__))
from htr_model.dataset import LMDBDataset, collate_fn, Converter
from htr_model.model_v2 import build_model_v2, IAM_ALPHABET
from confusion_spell import ConfusionSpellCorrector


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_cer(refs, hyps):
    total_dist, total_len = 0, 0
    for r, h in zip(refs, hyps):
        total_dist += editdistance.eval(r, h)
        total_len  += max(len(r), 1)
    return total_dist / total_len * 100


def build_vocab(lmdb_root='data/lmdb/train'):
    """Build vocabulary from IAM training LMDB labels."""
    import lmdb as _lmdb
    vocab = set()
    try:
        env = _lmdb.open(lmdb_root, readonly=True, lock=False)
        with env.begin() as txn:
            n = int(txn.get(b'num-samples').decode())
            for i in range(1, n + 1):
                lb = txn.get(f'label-{i:09d}'.encode())
                if lb:
                    for w in lb.decode().split():
                        vocab.add(w.strip('.,!?;:\'"()-[]'))
        env.close()
    except Exception as e:
        print(f'Warning: could not load vocab from LMDB: {e}')
    # Add words from system dictionary
    for path in ['/usr/share/dict/words', '/usr/dict/words']:
        if os.path.exists(path):
            with open(path) as f:
                vocab.update(w.strip() for w in f)
            break
    print(f'Vocabulary: {len(vocab):,} words')
    return vocab


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',     required=True)
    parser.add_argument('--split',          default='test')
    parser.add_argument('--lm_path',        default='lm/word_4gram.arpa')
    parser.add_argument('--lmdb_root',      default='data/lmdb')
    parser.add_argument('--confusion_csv',  required=True)
    parser.add_argument('--confusion_min',  type=int, default=5)
    parser.add_argument('--beam_width',     type=int, default=50)
    parser.add_argument('--alpha',          type=float, default=0.5)
    parser.add_argument('--beta',           type=float, default=1.0)
    parser.add_argument('--batch_size',     type=int, default=32)
    parser.add_argument('--run_name',       default='')
    parser.add_argument('--gpu',            type=int, default=0)
    parser.add_argument('--out_json',       default=None)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg  = ckpt.get('config', {})
    model = build_model_v2(
        IAM_ALPHABET,
        hidden_size=cfg.get('hidden_size', 512),
        dropout=cfg.get('dropout', 0.1),
    ).to(device)
    model.load_state_dict(ckpt.get('model', ckpt))
    model.eval()
    print(f'Loaded: {args.checkpoint}')

    # ── dataset ───────────────────────────────────────────────────────────────
    converter = Converter(IAM_ALPHABET)
    ds = LMDBDataset(os.path.join(args.lmdb_root, args.split))
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn,
                    num_workers=4, pin_memory=True)
    print(f'Dataset: {args.split} ({len(ds)} samples)')

    # ── beam decoder ──────────────────────────────────────────────────────────
    # Build mixed-case vocabulary
    import lmdb as _lmdb
    raw_vocab = set()
    for sub in ['train', 'val']:
        try:
            env = _lmdb.open(os.path.join(args.lmdb_root, sub), readonly=True, lock=False)
            with env.begin() as txn:
                n = int(txn.get(b'num-samples').decode())
                for i in range(1, n + 1):
                    lb = txn.get(f'label-{i:09d}'.encode())
                    if lb:
                        raw_vocab.update(lb.decode().split())
            env.close()
        except Exception:
            pass
    vocab_list = sorted(raw_vocab)
    decoder = build_ctcdecoder(
        labels=[''] + list(IAM_ALPHABET),
        kenlm_model_path=args.lm_path,
        alpha=args.alpha,
        beta=args.beta,
        unigrams=vocab_list,
    )
    print(f'Beam decoder ready (vocab={len(vocab_list):,})')

    # ── spell corrector ───────────────────────────────────────────────────────
    full_vocab = build_vocab(os.path.join(args.lmdb_root, 'train'))
    corrector = ConfusionSpellCorrector(
        confusion_csv=args.confusion_csv,
        vocabulary=full_vocab,
        min_count=args.confusion_min,
        max_ops=2,
    )

    # ── inference ─────────────────────────────────────────────────────────────
    refs, greedy_hyps, beam_hyps, spell_hyps = [], [], [], []

    with torch.no_grad():
        for imgs, labels_raw in dl:
            imgs = imgs.to(device)
            log_probs = model(imgs)                   # (T, B, C) — already log_softmax
            T, B, C = log_probs.shape

            # greedy — reuse converter.decode_batch
            greedy_hyps.extend(converter.decode_batch(log_probs))

            # beam
            probs_np = log_probs.permute(1, 0, 2).cpu().numpy()  # (B, T, C)
            for b in range(B):
                beam_out = decoder.decode(probs_np[b])
                beam_hyps.append(beam_out)
                spell_hyps.append(corrector.correct_line(beam_out))

            refs.extend(labels_raw)

    greedy_cer = compute_cer(refs, greedy_hyps)
    beam_cer   = compute_cer(refs, beam_hyps)
    spell_cer  = compute_cer(refs, spell_hyps)

    print(f'\n{"="*50}')
    print(f'Run: {args.run_name or args.checkpoint}')
    print(f'Greedy   CER = {greedy_cer:.2f}%')
    print(f'Beam     CER = {beam_cer:.2f}%   (Δ={beam_cer-greedy_cer:+.2f}%)')
    print(f'Spell    CER = {spell_cer:.2f}%   (Δ vs beam={spell_cer-beam_cer:+.2f}%)')
    print(f'{"="*50}')

    # sample output
    print('\n--- Changed samples (beam → spell) ---')
    shown = 0
    for r, b, s in zip(refs, beam_hyps, spell_hyps):
        if b != s and shown < 10:
            print(f'  REF  : {r}')
            print(f'  BEAM : {b}')
            print(f'  SPELL: {s}')
            cer_b = editdistance.eval(r, b) / max(len(r), 1) * 100
            cer_s = editdistance.eval(r, s) / max(len(r), 1) * 100
            print(f'  CER  : {cer_b:.1f}% → {cer_s:.1f}%  ({"✓ better" if cer_s < cer_b else "✗ worse"})')
            print()
            shown += 1

    if args.out_json:
        out = {
            'run_name':   args.run_name,
            'checkpoint': args.checkpoint,
            'confusion_csv': args.confusion_csv,
            'confusion_min': args.confusion_min,
            'n_samples':  len(refs),
            'greedy_cer': round(greedy_cer, 4),
            'beam_cer':   round(beam_cer,   4),
            'spell_cer':  round(spell_cer,  4),
            'beam_delta': round(beam_cer - greedy_cer, 4),
            'spell_delta': round(spell_cer - beam_cer, 4),
        }
        with open(args.out_json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Saved → {args.out_json}')


if __name__ == '__main__':
    main()
