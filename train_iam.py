"""
Train CRNN on IAM dataset (LMDB format).

Modes:
  original  – full training set
  clean     – flagged samples removed (train_clean LMDB)

Usage:
  python train_iam.py --mode original --run_name original
  python train_iam.py --mode clean    --run_name clean
  python train_iam.py --mode original --run_name original --gpu 0,1,2,3
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from htr_model.dataset import LMDBDataset, collate_fn, Converter


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir, run_name):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'train_{run_name}.log')
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger()


# ---------------------------------------------------------------------------
# CER / WER
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def evaluate(model, loader, converter, device):
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            log_probs = model(images)           # (T, N, C)
            decoded = converter.decode_batch(log_probs)
            refs.extend(labels)
            hyps.extend(decoded)
    model.train()
    return compute_cer_wer(refs, hyps)


# ---------------------------------------------------------------------------
# DataParallel wrapper: model outputs (T,N,C) but DP gathers on dim=0
# Wrapper transposes to (N,T,C) so DP can gather correctly along batch dim
# ---------------------------------------------------------------------------

class _DPWrapper(nn.Module):
    """Wraps CRNN so DataParallel gathers correctly along batch dim."""
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        return self.base(x).permute(1, 0, 2)   # (T,N,C) → (N,T,C)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',       choices=['original', 'clean'], default='original')
    parser.add_argument('--model',      choices=['v1', 'v2'], default='v1',
                        help='v1=BiLSTM (model.py), v2=BiGRU (model_v2.py)')
    parser.add_argument('--run_name',   default=None)
    parser.add_argument('--lmdb_dir',   default='./data/lmdb')
    parser.add_argument('--ckpt_dir',   default='./checkpoints')
    parser.add_argument('--log_dir',    default='./logs')
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--hidden',       type=int,   default=256)
    parser.add_argument('--dropout',      type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--workers',      type=int,   default=8)
    parser.add_argument('--gpu',        default='0',
                        help='GPU id(s), e.g. "0" or "0,1,2,3"')
    parser.add_argument('--resume',     default=None,
                        help='path to checkpoint .pt file to resume from (e.g. checkpoints/original_v2/ckpt_epoch80.pt)')
    parser.add_argument('--train_lmdb',  default=None,
                        help='override train LMDB path (bypasses --mode for train split)')
    parser.add_argument('--extra_train_lmdb', default=None,
                        help='additional synthetic LMDB to concat with IAM train')
    parser.add_argument('--synth_n',    type=int, default=None,
                        help='max synthetic samples to use (subset of extra_train_lmdb)')
    parser.add_argument('--val_lmdb',   default=None,
                        help='override validation LMDB path')
    args = parser.parse_args()

    # ---- GPU setup ----
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu_ids = list(range(len(args.gpu.split(','))))
    use_dp  = len(gpu_ids) > 1
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # batch_size is the TOTAL batch; DataParallel splits it across GPUs
    # This keeps gradient step count constant regardless of GPU count
    effective_batch = args.batch_size

    # ---- model factory ----
    if args.model == 'v2':
        from htr_model.model_v2 import build_model_v2, IAM_ALPHABET
        _build_model = lambda: build_model_v2(IAM_ALPHABET, hidden_size=args.hidden, dropout=args.dropout)
    else:
        from htr_model.model import build_model, IAM_ALPHABET
        _build_model = lambda: build_model(IAM_ALPHABET, hidden_size=args.hidden)

    run_name = args.run_name or f'{args.mode}_{args.model}'

    logger = setup_logging(args.log_dir, run_name)
    logger.info(f'Run: {run_name}  mode={args.mode}  device={device}  '
                f'gpus={args.gpu}  effective_batch={effective_batch}')

    # ---- datasets ----
    train_lmdb = args.train_lmdb or os.path.join(
        args.lmdb_dir, 'train_clean' if args.mode == 'clean' else 'train')
    val_lmdb = args.val_lmdb or os.path.join(
        args.lmdb_dir, 'val_clean' if args.mode == 'clean' else 'val')

    from torch.utils.data import ConcatDataset, Subset
    train_ds = LMDBDataset(train_lmdb, augment=True)
    if args.extra_train_lmdb:
        synth_ds = LMDBDataset(args.extra_train_lmdb, augment=True)
        if args.synth_n and args.synth_n < len(synth_ds):
            import random as _rnd
            # Fixed seed based on synth_n so all experiments at the same scale
            # use IDENTICAL synthetic samples regardless of mode (full/clean).
            _rng = _rnd.Random(42 + args.synth_n)
            idx = _rng.sample(range(len(synth_ds)), args.synth_n)
            synth_ds = Subset(synth_ds, sorted(idx))
            logger.info(f'Synth subset seed=42+{args.synth_n}, indices fixed')
        train_ds = ConcatDataset([train_ds, synth_ds])
        logger.info(f'Synth added: {len(synth_ds)} samples  '
                    f'(from {args.extra_train_lmdb})')
    val_ds = LMDBDataset(val_lmdb, augment=False)
    logger.info(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=effective_batch,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=effective_batch,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=args.workers, pin_memory=True)

    # ---- model ----
    converter = Converter(IAM_ALPHABET)
    base_model = _build_model().to(device)
    total_p = sum(p.numel() for p in base_model.parameters())
    logger.info(f'Model params: {total_p:,}  DataParallel={use_dp}  gpus={gpu_ids}')

    if use_dp:
        wrapped = _DPWrapper(base_model)
        model   = nn.DataParallel(wrapped, device_ids=gpu_ids)
        # eval uses base_model directly (single-GPU, no wrapper overhead)
        eval_model = base_model
    else:
        model      = base_model
        eval_model = base_model

    # ---- optimizer & scheduler ----
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Phase 1: Linear warmup (step-level, epochs 1~warmup_epochs)
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    warmup_steps    = args.warmup_epochs * steps_per_epoch
    warmup_sched    = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6 / args.lr,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # Phase 2: ReduceLROnPlateau (epoch-level, monitors val_CER)
    # patience=10: with augmentation, val_CER can fluctuate ±1% naturally;
    # too-small patience causes premature LR decay and stalls training
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=10,
        verbose=True,
        threshold=0.001,
        threshold_mode='rel',
        cooldown=2,
        min_lr=1e-7,
        eps=1e-7,
    )

    # CTC loss: mean reduction (normalizes by target length + batch)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    # ---- training loop ----
    ckpt_dir = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_cer   = float('inf')
    history    = []
    start_epoch = 1

    # ---- resume from checkpoint ----
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        base_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        plateau_sched.load_state_dict(ckpt['plateau_sched'])
        start_epoch = ckpt['epoch'] + 1
        best_cer    = ckpt.get('best_cer', float('inf'))
        history     = ckpt.get('history', [])
        logger.info(f'Resumed from {args.resume}  epoch={ckpt["epoch"]}  best_cer={best_cer:.2f}%')

    for epoch in range(start_epoch, args.epochs + 1):
        if use_dp:
            model.train()
        else:
            base_model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            targets, target_lens = converter.encode(labels)
            targets     = targets.to(device)
            target_lens = target_lens.to(device)

            if use_dp:
                # model returns (N, T, C); transpose for CTC
                out = model(images)                     # (N, T, C)
                log_probs = out.permute(1, 0, 2)        # (T, N, C)
            else:
                log_probs = model(images)               # (T, N, C)

            T, N, _ = log_probs.shape
            input_lens = torch.full((N,), T, dtype=torch.long, device=device)

            loss = criterion(log_probs, targets, input_lens, target_lens)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(base_model.parameters(), 10.0)
            optimizer.step()

            epoch_loss += loss.item()
            if epoch <= args.warmup_epochs:
                warmup_sched.step()   # warmup: step per batch

            if step % 50 == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info(f'Epoch {epoch}/{args.epochs}  step {step}/{len(train_loader)}'
                            f'  loss={loss.item():.4f}  lr={lr:.2e}')

        cer, wer = evaluate(eval_model, val_loader, converter, device)
        elapsed = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)

        # Plateau scheduler: step after warmup, using val_CER as metric
        if epoch > args.warmup_epochs:
            plateau_sched.step(cer)

        lr = optimizer.param_groups[0]['lr']
        logger.info(f'[Epoch {epoch}] loss={avg_loss:.4f}  '
                    f'val_CER={cer:.2f}%  val_WER={wer:.2f}%  '
                    f'lr={lr:.2e}  time={elapsed:.0f}s')

        history.append({'epoch': epoch, 'loss': avg_loss, 'cer': cer, 'wer': wer})

        if cer < best_cer:
            best_cer = cer
            torch.save(base_model.state_dict(), os.path.join(ckpt_dir, 'best.pt'))
            logger.info(f'  ↳ New best val CER={cer:.2f}% → saved best.pt')

        # Full checkpoint (model + optimizer + scheduler) for resume
        full_ckpt = {
            'epoch':        epoch,
            'model':        base_model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'plateau_sched': plateau_sched.state_dict(),
            'best_cer':     best_cer,
            'history':      history,
        }
        torch.save(full_ckpt, os.path.join(ckpt_dir, 'latest_ckpt.pt'))
        # Keep a snapshot every 20 epochs for safety
        if epoch % 20 == 0:
            torch.save(full_ckpt, os.path.join(ckpt_dir, f'ckpt_epoch{epoch:03d}.pt'))
            logger.info(f'  ↳ Checkpoint saved: ckpt_epoch{epoch:03d}.pt')

    # save curve
    curve_path = os.path.join(ckpt_dir, 'training_curve.json')
    with open(curve_path, 'w') as f:
        json.dump({'run': run_name, 'best_cer': best_cer, 'epochs': history}, f, indent=2)

    logger.info(f'Done. Best val CER: {best_cer:.2f}%  →  {ckpt_dir}/best.pt')


if __name__ == '__main__':
    main()
