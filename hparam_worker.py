"""
Single-experiment worker for parallel hparam search.

Usage:
    CUDA_VISIBLE_DEVICES=2 python hparam_worker.py --run_id 3 --config '{"dropout":0.0,...}'

Writes result to results/hparam_search.tsv (with file locking).
"""

import argparse
import fcntl
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

# -----------------------------------------------------------------------
LMDB_TRAIN = './data/lmdb/train'
LMDB_VAL   = './data/lmdb/val'
EPOCHS      = 50
WARMUP      = 1
WORKERS     = 4
TSV_PATH    = './results/hparam_search.tsv'
CKPT_DIR    = './checkpoints/hparam_search'
LOG_DIR     = './logs/hparam_workers'

os.makedirs('./results', exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def edit_distance(a, b):
    import editdistance
    return editdistance.eval(a, b)

def compute_cer(refs, hyps):
    errs = total = 0
    for r, h in zip(refs, hyps):
        errs  += edit_distance(r, h)
        total += max(len(r), 1)
    return errs / total * 100

def evaluate(model, loader, converter, device):
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            log_probs = model(images)
            hyps.extend(converter.decode_batch(log_probs))
            refs.extend(labels)
    model.train()
    return compute_cer(refs, hyps)


def build_model_v2_custom(alphabet, cfg):
    import torch.nn as nn
    import torch.nn.functional as F
    from htr_model.model_v2 import Encoder, CRNN_v2

    num_classes = len(alphabet) + 1

    if cfg.get('use_original_vgg', False):
        class OriginalVGG(nn.Module):
            def __init__(self, input_channel=1, output_channel=512):
                super().__init__()
                ks = [3, 3, 3, 3, 3, 3, 2]
                ps = [1, 1, 1, 1, 1, 1, 0]
                ss = [1, 1, 1, 1, 1, 1, 1]
                nm = [64, 128, 256, 256, 512, 512, output_channel]
                cnn = nn.Sequential()
                def convRelu(i, bn=False):
                    nIn  = input_channel if i == 0 else nm[i-1]
                    nOut = nm[i]
                    cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
                    if bn: cnn.add_module(f'bn{i}', nn.BatchNorm2d(nOut))
                    cnn.add_module(f'relu{i}', nn.ReLU(True))
                convRelu(0)
                cnn.add_module('pool0', nn.MaxPool2d(2, 2))
                convRelu(1)
                cnn.add_module('pool1', nn.MaxPool2d(2, 2))
                convRelu(2, True); convRelu(3)
                cnn.add_module('pool2', nn.MaxPool2d((2,2),(2,1),(0,1)))
                convRelu(4, True); convRelu(5)
                cnn.add_module('pool3', nn.MaxPool2d((2,2),(2,1),(0,1)))
                convRelu(6, True)
                self.ConvNet = cnn
            def forward(self, x):
                return self.ConvNet(x)

        class CRNNOrigVGG(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn     = OriginalVGG(1, cfg['cnn_out'])
                self.dropout = nn.Dropout(cfg['dropout'])
                self.encoder = Encoder(cfg['cnn_out'], cfg['hidden'], 2, True, cfg['dropout'])
                self.decoder = nn.Linear(cfg['hidden'] * 2, num_classes)
            def forward(self, x):
                import torch.nn.functional as F
                feat = self.cnn(x).mean(dim=2).permute(0,2,1)
                feat = self.dropout(feat)
                out, _ = self.encoder(feat)
                out = self.decoder(out).permute(1,0,2)
                return F.log_softmax(out, dim=2)

        return CRNNOrigVGG()

    return CRNN_v2(
        img_h=64, num_channels=1, num_classes=num_classes,
        encoder_input_size=cfg['cnn_out'],
        encoder_hidden_size=cfg['hidden'],
        encoder_layers=2, encoder_bidirectional=True,
        dropout=cfg['dropout'],
    )


def log_tsv_locked(run_id, cer, status, cfg):
    config_str = json.dumps({k: v for k, v in cfg.items() if k != 'desc'})
    line = f'{run_id:02d}\t{cer:.4f}\t{status}\t{cfg["desc"]}\t{config_str}\n'
    with open(TSV_PATH, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_experiment(cfg, run_id, device, logger):
    from htr_model.model_v2 import build_model_v2, IAM_ALPHABET

    logger.info(f'Run {run_id:02d}: {cfg["desc"]}')
    logger.info(f'Config: {cfg}')

    train_ds = LMDBDataset(LMDB_TRAIN, augment=True)
    val_ds   = LMDBDataset(LMDB_VAL,   augment=False)

    bs = cfg['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=collate_fn, num_workers=WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=WORKERS, pin_memory=True)

    converter = Converter(IAM_ALPHABET)
    if cfg['cnn_out'] != 512 or cfg.get('use_original_vgg', False):
        model = build_model_v2_custom(IAM_ALPHABET, cfg)
    else:
        model = build_model_v2(
            IAM_ALPHABET,
            hidden_size=cfg['hidden'],
            dropout=cfg['dropout'],
        )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    steps_per_epoch = max(1, len(train_ds) // bs)
    warmup_steps    = WARMUP * steps_per_epoch
    warmup_sched    = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6/cfg['lr'], end_factor=1.0, total_iters=warmup_steps)
    plateau_sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10, threshold=0.001,
        cooldown=2, min_lr=1e-7, eps=1e-7)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    best_cer = float('inf')
    t_start  = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            targets, target_lens = converter.encode(labels)
            targets     = targets.to(device)
            target_lens = target_lens.to(device)
            log_probs   = model(images)
            T, N, _     = log_probs.shape
            input_lens  = torch.full((N,), T, dtype=torch.long, device=device)
            loss        = criterion(log_probs, targets, input_lens, target_lens)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            if epoch <= WARMUP:
                warmup_sched.step()

        cer = evaluate(model, val_loader, converter, device)
        if epoch > WARMUP:
            plateau_sched.step(cer)

        if cer < best_cer:
            best_cer = cer
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, f'run{run_id:02d}_best.pt'))

        if epoch % 10 == 0 or epoch == EPOCHS:
            elapsed = time.time() - t_start
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'  Epoch {epoch:3d}/{EPOCHS}  val_CER={cer:.2f}%  '
                        f'best={best_cer:.2f}%  lr={lr:.2e}  t={elapsed:.0f}s')

    logger.info(f'Run {run_id:02d} DONE  best_val_CER={best_cer:.2f}%  ({cfg["desc"]})')
    return best_cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id',   type=int, required=True)
    parser.add_argument('--config',   type=str, required=True,
                        help='JSON string with full config (including desc)')
    parser.add_argument('--baseline_cer', type=float, required=True,
                        help='CER of current best config (for keep/discard decision)')
    args = parser.parse_args()

    cfg = json.loads(args.config)

    log_path = os.path.join(LOG_DIR, f'run{args.run_id:02d}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Run {args.run_id:02d} device: {device}  GPU: {os.environ.get("CUDA_VISIBLE_DEVICES","?")}')

    cer = run_experiment(cfg, args.run_id, device, logger)

    if cer < args.baseline_cer - 0.05:
        status = 'keep'
        logger.info(f'  KEEP  {cer:.2f}% < {args.baseline_cer:.2f}% (baseline)')
    else:
        status = 'discard'
        logger.info(f'  DISCARD  {cer:.2f}% >= {args.baseline_cer:.2f}% (baseline)')

    log_tsv_locked(args.run_id, cer, status, cfg)
    logger.info(f'Result written to {TSV_PATH}')


if __name__ == '__main__':
    main()
