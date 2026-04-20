"""
Hyperparameter search for CRNN/IAM — autoresearch style.

Rules (from autoresearch/program.md):
- Fixed 50-epoch budget per experiment
- One change at a time from current best config
- Keep if val_CER improves, discard otherwise
- Log everything to hparam_search.tsv
- NEVER STOP until all experiments exhausted

Metric: best val_CER over 50 epochs (lower is better).
"""

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
# Fixed constants
# -----------------------------------------------------------------------
LMDB_TRAIN = './data/lmdb/train'
LMDB_VAL   = './data/lmdb/val'
EPOCHS      = 50
WARMUP      = 1
GPU         = '0'
WORKERS     = 8
TSV_PATH    = './results/hparam_search.tsv'
CKPT_DIR    = './checkpoints/hparam_search'
LOG_PATH    = './logs/hparam_search.log'

os.makedirs('./results', exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs('./logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger()


# -----------------------------------------------------------------------
# Experiment definitions — each is a delta from baseline
# -----------------------------------------------------------------------
BASELINE = dict(
    hidden=256,
    dropout=0.1,
    batch_size=64,
    lr=1e-4,
    cnn_out=512,           # encoder_input_size
    weight_decay=1e-4,
    use_original_vgg=False, # False = our H=64 VGG (last_kernel=4)
    desc='baseline: hidden=256 dropout=0.1 bs=64 lr=1e-4',
)

EXPERIMENTS = [
    # --- hidden size ---
    dict(hidden=128,  desc='hidden=128'),
    dict(hidden=512,  desc='hidden=512'),
    # --- dropout ---
    dict(dropout=0.0, desc='dropout=0.0 (no dropout)'),
    dict(dropout=0.2, desc='dropout=0.2'),
    dict(dropout=0.3, desc='dropout=0.3'),
    # --- batch size ---
    dict(batch_size=32,  desc='batch_size=32'),
    dict(batch_size=128, desc='batch_size=128'),
    # --- learning rate ---
    dict(lr=3e-4, desc='lr=3e-4'),
    dict(lr=5e-5, desc='lr=5e-5'),
    # --- CNN output channels ---
    dict(cnn_out=256, desc='cnn_out=256 (smaller CNN)'),
    # --- weight decay ---
    dict(weight_decay=1e-3, desc='weight_decay=1e-3'),
    dict(weight_decay=0.0,  desc='weight_decay=0 (no wd)'),
    # --- CNN architecture: original VGG (last kernel=2, H=32 style) vs ours (kernel=4, H=64) ---
    dict(use_original_vgg=True, desc='original_vgg (last_kernel=2, H=32 style)'),
]


# -----------------------------------------------------------------------
# CER helpers
# -----------------------------------------------------------------------
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


# -----------------------------------------------------------------------
# Single training run — returns best val_CER over all epochs
# -----------------------------------------------------------------------
def run_experiment(cfg, run_id, device):
    from htr_model.model_v2 import build_model_v2, IAM_ALPHABET

    logger.info(f'\n{"="*60}')
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


def build_model_v2_custom(alphabet, cfg):
    """Build model with non-default structural params."""
    import torch.nn as nn
    import torch.nn.functional as F
    from htr_model.model_v2 import Encoder, CRNN_v2

    num_classes = len(alphabet) + 1

    if cfg.get('use_original_vgg', False):
        # Original VGG_FeatureExtractor: last kernel=2 (designed for H=32)
        # With H=64 input, last conv outputs h=2 instead of h=1
        # We handle this with mean(dim=2) in forward — same as CRNN_v2
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


# -----------------------------------------------------------------------
# TSV logging
# -----------------------------------------------------------------------
def init_tsv():
    if not os.path.exists(TSV_PATH):
        with open(TSV_PATH, 'w') as f:
            f.write('run_id\tval_cer\tstatus\tdescription\tconfig\n')

def log_tsv(run_id, cer, status, cfg):
    config_str = json.dumps({k: v for k, v in cfg.items() if k != 'desc'})
    with open(TSV_PATH, 'a') as f:
        f.write(f'{run_id:02d}\t{cer:.4f}\t{status}\t{cfg["desc"]}\t{config_str}\n')


# -----------------------------------------------------------------------
# Main search loop
# -----------------------------------------------------------------------
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    init_tsv()

    # ---- Resume from TSV if it already has entries ----
    # Determine which run to start from and what the current best config is
    import csv
    completed_runs = {}
    if os.path.exists(TSV_PATH):
        with open(TSV_PATH) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                completed_runs[int(row['run_id'])] = row

    if 0 not in completed_runs:
        current_best = dict(BASELINE)
        run_id = 0
        logger.info('=== RUN 00: BASELINE ===')
        baseline_cer = run_experiment(current_best, run_id, device)
        log_tsv(run_id, baseline_cer, 'keep', current_best)
        logger.info(f'Baseline val_CER = {baseline_cer:.2f}%')
    else:
        # Find current best from completed runs
        kept = [r for r in completed_runs.values() if r['status'] == 'keep']
        best_row = min(kept, key=lambda r: float(r['val_cer']))
        baseline_cer = float(best_row['val_cer'])
        current_best = dict(BASELINE)
        current_best.update(json.loads(best_row['config']))
        run_id = max(completed_runs.keys())
        logger.info(f'Resuming from run {run_id}, current best={baseline_cer:.4f}%')
        logger.info(f'Current best config: {current_best}')

    # Iterate over experiments — skip already completed ones
    for exp_delta in EXPERIMENTS:
        run_id += 1
        if run_id in completed_runs:
            logger.info(f'Skipping run {run_id:02d} (already in TSV)')
            # Update current_best if this run was kept
            row = completed_runs[run_id]
            if row['status'] == 'keep':
                baseline_cer = float(row['val_cer'])
                current_best.update(json.loads(row['config']))
            continue

        # Build candidate config from current best + delta
        candidate = dict(current_best)
        candidate.update(exp_delta)

        cer = run_experiment(candidate, run_id, device)

        if cer < baseline_cer - 0.05:   # must improve by >0.05% to keep
            logger.info(f'  KEEP  {cer:.2f}% < {baseline_cer:.2f}% (prev best)')
            log_tsv(run_id, cer, 'keep', candidate)
            current_best  = dict(candidate)
            baseline_cer  = cer
        else:
            logger.info(f'  DISCARD  {cer:.2f}% >= {baseline_cer:.2f}% (prev best)')
            log_tsv(run_id, cer, 'discard', candidate)

    # Summary
    logger.info('\n' + '='*60)
    logger.info('SEARCH COMPLETE')
    logger.info(f'Best config: {current_best}')
    logger.info(f'Best val_CER: {baseline_cer:.2f}%')
    with open('./results/hparam_best_config.json', 'w') as f:
        json.dump({'best_cer': baseline_cer, 'config': current_best}, f, indent=2)


if __name__ == '__main__':
    main()
