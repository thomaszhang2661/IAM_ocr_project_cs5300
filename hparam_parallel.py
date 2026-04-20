"""
Parallel hyperparameter search launcher.

Reads completed runs from hparam_search.tsv, then launches remaining
experiments across available GPUs using hparam_worker.py.

All experiments use the current best config (from TSV) as the base.
Runs in batches — each GPU processes one experiment at a time.

Usage:
    python hparam_parallel.py [--gpus 0,2,3,4,5,6,7]
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

# -----------------------------------------------------------------------
TSV_PATH  = './results/hparam_search.tsv'
LOG_PATH  = './logs/hparam_parallel.log'

os.makedirs('./logs', exist_ok=True)
os.makedirs('./results', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger()

# -----------------------------------------------------------------------
BASELINE_FULL = dict(
    hidden=256, dropout=0.1, batch_size=64, lr=1e-4,
    cnn_out=512, weight_decay=1e-4, use_original_vgg=False,
    desc='baseline: hidden=256 dropout=0.1 bs=64 lr=1e-4',
)

# All experiments with their canonical run_id
ALL_EXPERIMENTS = [
    (0,  dict(desc='baseline: hidden=256 dropout=0.1 bs=64 lr=1e-4')),   # run 00
    (1,  dict(hidden=128,  desc='hidden=128')),
    (2,  dict(hidden=512,  desc='hidden=512')),
    (3,  dict(dropout=0.0, desc='dropout=0.0 (no dropout)')),
    (4,  dict(dropout=0.2, desc='dropout=0.2')),
    (5,  dict(dropout=0.3, desc='dropout=0.3')),
    (6,  dict(batch_size=32,  desc='batch_size=32')),
    (7,  dict(batch_size=128, desc='batch_size=128')),
    (8,  dict(lr=3e-4, desc='lr=3e-4')),
    (9,  dict(lr=5e-5, desc='lr=5e-5')),
    (10, dict(cnn_out=256, desc='cnn_out=256 (smaller CNN)')),
    (11, dict(weight_decay=1e-3, desc='weight_decay=1e-3')),
    (12, dict(weight_decay=0.0,  desc='weight_decay=0 (no wd)')),
    (13, dict(use_original_vgg=True, desc='original_vgg (last_kernel=2, H=32 style)')),
]


def read_completed():
    """Return dict run_id -> row from TSV."""
    completed = {}
    if not os.path.exists(TSV_PATH):
        return completed
    with open(TSV_PATH) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            completed[int(row['run_id'])] = row
    return completed


def best_config_from_tsv(completed):
    """Return (best_cer, best_full_config) from kept runs."""
    kept = [r for r in completed.values() if r['status'] == 'keep']
    if not kept:
        return None, None
    best_row = min(kept, key=lambda r: float(r['val_cer']))
    cfg = dict(BASELINE_FULL)
    cfg.update(json.loads(best_row['config']))
    return float(best_row['val_cer']), cfg


def launch_worker(run_id, cfg, baseline_cer, gpu_id, conda_env='ocr_IAM'):
    """Launch hparam_worker.py as a subprocess on the given GPU."""
    config_json = json.dumps(cfg)
    cmd = (
        f'conda run -n {conda_env} --no-capture-output '
        f'python hparam_worker.py '
        f'--run_id {run_id} '
        f'--baseline_cer {baseline_cer} '
        f'--config \'{config_json}\''
    )
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    log_file = open(f'./logs/hparam_workers/launcher_run{run_id:02d}.log', 'w')
    proc = subprocess.Popen(
        cmd, shell=True, env=env,
        stdout=log_file, stderr=log_file,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return proc, log_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0,2,3,4,5,6,7',
                        help='Comma-separated GPU indices to use')
    parser.add_argument('--conda_env', default='ocr_IAM')
    args = parser.parse_args()

    gpu_list = [int(g) for g in args.gpus.split(',')]
    logger.info(f'Available GPUs: {gpu_list}')

    completed = read_completed()
    logger.info(f'Completed runs: {sorted(completed.keys())}')

    # Determine base config and CER
    baseline_cer, current_best = best_config_from_tsv(completed)
    if current_best is None:
        baseline_cer = float('inf')
        current_best = dict(BASELINE_FULL)
    logger.info(f'Base config (from best kept run): CER={baseline_cer:.4f}%')
    logger.info(f'Base config: {current_best}')

    # Determine which experiments still need to run
    pending = []
    for run_id, delta in ALL_EXPERIMENTS:
        if run_id in completed:
            logger.info(f'  Skip run {run_id:02d} (already done, cer={completed[run_id]["val_cer"]})')
            continue
        cfg = dict(current_best)
        cfg.update(delta)
        pending.append((run_id, cfg))

    if not pending:
        logger.info('All experiments already completed!')
        _print_summary(completed)
        return

    logger.info(f'\n{len(pending)} experiments to run: {[r for r, _ in pending]}')
    logger.info(f'Distributing across {len(gpu_list)} GPUs\n')

    # Round-robin assignment: use a pool of (gpu_id, process) slots
    gpu_pool   = list(gpu_list)   # free GPU queue
    running    = {}               # gpu_id -> (proc, log_file, run_id)
    pending_q  = list(pending)

    os.makedirs('./logs/hparam_workers', exist_ok=True)

    def poll_running():
        """Check for finished processes, free their GPUs."""
        done_gpus = []
        for gpu_id, (proc, lf, rid) in running.items():
            if proc.poll() is not None:
                lf.close()
                rc = proc.returncode
                status = 'OK' if rc == 0 else f'FAILED(rc={rc})'
                logger.info(f'  GPU {gpu_id}: run {rid:02d} finished — {status}')
                done_gpus.append(gpu_id)
        for g in done_gpus:
            gpu_pool.append(g)
            del running[g]
        return len(done_gpus) > 0

    # Dispatch loop
    while pending_q or running:
        # Launch as many as we have free GPUs
        while pending_q and gpu_pool:
            gpu_id = gpu_pool.pop(0)
            run_id, cfg = pending_q.pop(0)
            logger.info(f'  Launching run {run_id:02d} on GPU {gpu_id}: {cfg["desc"]}')
            proc, lf = launch_worker(run_id, cfg, baseline_cer, gpu_id, args.conda_env)
            running[gpu_id] = (proc, lf, run_id)

        time.sleep(10)
        poll_running()

    logger.info('\nAll experiments finished. Collecting results...')
    time.sleep(2)  # let TSV writes settle

    completed = read_completed()
    _print_summary(completed)


def _print_summary(completed):
    logger.info('\n' + '='*60)
    logger.info('HPARAM SEARCH SUMMARY')
    logger.info('='*60)
    rows = sorted(completed.values(), key=lambda r: int(r['run_id']))
    for r in rows:
        flag = '  *** KEEP ***' if r['status'] == 'keep' else ''
        logger.info(f'  Run {r["run_id"]}  {float(r["val_cer"]):.2f}%  [{r["status"]}]  {r["description"]}{flag}')

    kept = [r for r in completed.values() if r['status'] == 'keep']
    if kept:
        best = min(kept, key=lambda r: float(r['val_cer']))
        logger.info(f'\nBest: run {best["run_id"]}  {float(best["val_cer"]):.2f}%  {best["description"]}')
        best_cfg = dict(BASELINE_FULL)
        best_cfg.update(json.loads(best['config']))
        logger.info(f'Best config: {best_cfg}')

        import json as _json
        with open('./results/hparam_best_config.json', 'w') as f:
            _json.dump({'best_cer': float(best['val_cer']), 'config': best_cfg}, f, indent=2)
        logger.info('Saved to results/hparam_best_config.json')


if __name__ == '__main__':
    main()
