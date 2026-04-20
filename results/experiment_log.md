# Experiment Log — IAM Handwriting Recognition with VLM-Cleaned Labels

**Project:** CS5300 Final Project  
**Goal:** Compare CRNN-CTC model performance under four training conditions:
1. Full IAM labels → test
2. VLM-cleaned IAM labels → test
3. Full IAM + synthetic data → test
4. Cleaned IAM + synthetic data → test

---

## Dataset

### IAM Handwriting Database (HF version)
| Split | Samples |
|-------|---------|
| train | 6,482   |
| val   | 976     |
| test  | 2,915   |

### VLM Annotation Cleaning (Doubao / ByteDance)
- **Model:** `ep-20260214152858-8r9sn` (Doubao VLM)
- **API:** `https://ark.cn-beijing.volces.com/api/v3` (OpenAI-compatible)
- **Task:** For each sample, check whether the ground-truth label matches the handwritten image; if incorrect, provide the corrected text.

| Split | Total  | Correct (%)  | Flagged (%) |
|-------|--------|-------------|-------------|
| train | 6,482  | 4,336 (66.9%) | 2,144 (33.1%) |
| val   | 976    | 616 (63.1%)   | 360 (36.9%)   |
| test  | 2,915  | 1,862 (63.9%) | 1,053 (36.1%) |

Flagged samples were **removed** (not relabeled) to build clean LMDB splits:
- `data/lmdb/train_clean` — 4,338 samples
- `data/lmdb/val_clean`   — 616 samples

---

## Model Architecture

**CRNN-CTC** (model_v2):
- **CNN backbone:** Modified VGG-style (6 conv layers, BatchNorm, 2×pooling in width only to preserve height)
- **RNN:** 2-layer BiGRU, hidden_size=512
- **Output:** Linear → CTC loss (81 classes = 80 IAM chars + blank)
- **Input:** grayscale 32×256 images, normalized to [-1, 1]
- **Total params:** ~16.7M

---

## Phase 1 — Hyperparameter Search

### Method
- **Budget:** 50 epochs per run (proxy for full 150-epoch training)
- **Strategy:** One-change-at-a-time greedy search (autoresearch style)
- **Baseline:** hidden=256, dropout=0.1, bs=64, lr=1e-4
- **Accept rule:** keep if new val CER < current best (strict improvement)
- **Execution:** 7 GPUs in parallel via `hparam_parallel.py` + `hparam_worker.py`

### Results (all 14 runs)

| Run | Val CER | Decision | Change |
|-----|---------|----------|--------|
| 00 | 9.04% | keep | baseline |
| 01 | 9.79% | discard | hidden=128 |
| **02** | **7.35%** | **keep** | **hidden=512** |
| 03 | 7.45% | discard | dropout=0.0 |
| 04 | 7.85% | discard | dropout=0.2 |
| 05 | 7.20% | keep | dropout=0.3 |
| 06 | 6.73% | keep | batch_size=32 |
| 07 | 8.39% | discard | batch_size=128 |
| **08** | **6.29%** | **keep** | **lr=3e-4** |
| 09 | 8.99% | discard | lr=5e-5 |
| 10 | 7.50% | discard | cnn_out=256 |
| 11 | 8.05% | discard | weight_decay=1e-3 |
| 12 | 7.42% | discard | weight_decay=0 |
| 13 | 7.45% | discard | original_vgg architecture |

### Best Configuration (saved: `results/hparam_best_config.json`)
```json
{
  "hidden": 512, "lr": 3e-4, "dropout": 0.1,
  "batch_size": 64, "cnn_out": 512,
  "weight_decay": 1e-4, "use_original_vgg": false
}
```
**Best proxy val CER: 6.29%** (50-epoch run08)

**Preliminary test evaluation on run08 best checkpoint:**
- Greedy CER: 9.58%,  WER: 29.72%
- Beam search (w=50, α=0.5, β=1.0): CER 9.50%, WER 29.49%

---

## Phase 2 — Formal Training Experiments

### Shared Settings
- Epochs: 150
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
- LR schedule: 1-epoch LinearLR warmup → ReduceLROnPlateau (patience=10, factor=0.8, min_lr=1e-7)
- Batch size: 64
- Augmentation: random rotation ±3°, width jitter ±5%, Gaussian noise (train only)
- Checkpointing: `ckpt_epoch020.pt`, `ckpt_epoch040.pt`, ..., `latest_ckpt.pt`, `best.pt`

---

### Exp1 — Full IAM

| Setting | Value |
|---------|-------|
| Training data | `data/lmdb/train` (6,482 samples) |
| Validation data | `data/lmdb/val` (976 samples) |
| GPU | 0 (H100 80GB) |
| Checkpoint dir | `checkpoints/exp1_full_iam/` |
| Log | `logs/train_exp1_full_iam.log` |
| History CSV | `results/exp1_full_iam_history.csv` |

**Progress:** Training resumed from epoch 45. Current best val CER: **~5.75%** (epoch 85).  
*(Full 150-epoch results TBD)*

---

### Exp2 — VLM-Cleaned IAM

| Setting | Value |
|---------|-------|
| Training data | `data/lmdb/train_clean` (4,338 samples, 2,144 removed) |
| Validation data | `data/lmdb/val_clean` (616 samples, 360 removed) |
| GPU | 2 (H100 80GB) |
| Checkpoint dir | `checkpoints/exp2_clean_iam/` |
| Log | `logs/train_exp2_clean_iam.log` (new run starts at line 304, Val: 616) |
| History CSV | `results/exp2_clean_iam_history.csv` |

**Note:** Val set also cleaned to ensure fair, unbiased evaluation.  
**Progress:** Restarted from epoch 1 with val_clean. Current epoch ~36, val CER ~9.58%.  
*(Full 150-epoch results TBD)*

---

### Exp3 — Full IAM + Synthetic Data (planned)

| Scale | Synthetic samples | Total train |
|-------|------------------|-------------|
| 1×   | 6,482            | 12,964      |
| 2×   | 12,964           | 19,446      |
| 3×   | 19,446           | 25,928      |
| 5×   | 32,410           | 38,892      |
| 7×   | 45,374           | 51,856      |
| 10×  | 64,820           | 71,302      |

Synthetic data generation: `data/generate_synthetic.py`  
Text pool: IAM labels × 2 + Doubao-generated rare-char sentences × 3 + targeted patterns × 2  
*(Results TBD)*

---

### Exp4 — Cleaned IAM + Synthetic Data (planned)

Same scales as Exp3 but using `train_clean` as base.  
*(Results TBD)*

---

## Phase 3 — Post-Processing (Decoding)

### Beam Search (pyctcdecode)
- Labels: `[''] + list(IAM_ALPHABET)` (blank at index 0)
- Vocabulary: IAM train+val words + `/usr/share/dict/words` → 73,540 unique words
- Config: beam_width=50, α=0.5 (LM weight), β=1.0 (word insertion bonus)
- KenLM: not available (build failed, documented as future work)

**Result on run08 checkpoint (50-epoch proxy model):**
- Greedy → Beam improvement: CER +0.08%, WER +0.23%
- Expected larger improvement on full 150-epoch models

### Confusion-Based Spell Corrector (`confusion_spell.py`)
- Input: `results/vlm_confusion_all.csv` (2,846 confusion pairs)
- Operations: 1:1 substitutions, n:1 merges, 1:n splits (symmetric)
- min_count threshold: 5
- Policy: correct only if **unique** vocabulary match (avoids ambiguous corrections)
- 2-op corrections: enabled with cnt≥10 threshold

---

## VLM Confusion Analysis

### Files
| File | Description |
|------|-------------|
| `results/vlm_confusion_all.csv` | All 2,846 confusion pairs (sub/merge/split/insert/delete) |
| `results/vlm_confusion_matrix_1to1.png` | 26×26 heatmap, lowercase 1:1 substitutions |
| `results/vlm_confusion_top35.png` | Top-35 bar chart (sub=red, merge=purple, split=blue) |
| `results/vlm_confusion_multichar.png` | Merge & split pairs separately |
| `results/vlm_confusion_report.html` | Interactive HTML with example images |

### Top Confusion Types
| Type | Pairs | Total events |
|------|-------|-------------|
| substitute_1:1 | 652 | ~15,000+ |
| merge_n:1 | ~80 | ~500 |
| split_1:n | ~90 | ~500 |
| delete_n:0 | ~200 | ~800 |
| insert_0:n | ~200 | ~600 |

Key 1:1 pairs: `n→m` (99×), `r→s` (95×), `u→n` (88×), `h→b` (79×)  
Key merges: `th→H` (16×), `cl→d` (15×)  
Key splits: `d→cl` (15×), `d→ch` (11×)

---

## Key Files Reference

```
final_project/
├── train_iam.py                 # Main training script
├── htr_model/
│   ├── model_v2.py              # CRNN architecture
│   └── dataset.py               # LMDBDataset, augmentation
├── decode_beam.py               # Beam search decoding
├── confusion_spell.py           # Confusion-based spell corrector
├── analyze_vlm_confusion.py     # 1:1 confusion analysis (char-level)
├── analyze_vlm_confusion_v2.py  # Extended: merge/split/insert/delete
├── hparam_worker.py             # Single hparam run worker
├── hparam_parallel.py           # Multi-GPU hparam launcher
├── data/
│   ├── prepare_lmdb.py          # Build train/val/test LMDBs
│   ├── create_val_clean_lmdb.py # Build val_clean LMDB
│   ├── generate_synthetic.py    # Synthetic data generation (trdg)
│   └── generate_rare_texts.py   # Doubao API rare-char sentence generator
├── checkpoints/
│   ├── exp1_full_iam/           # Full IAM model weights
│   ├── exp2_clean_iam/          # Cleaned IAM model weights
│   └── hparam_search/           # run08_best.pt (best 50-epoch model)
├── results/
│   ├── hparam_search.tsv        # All 14 hparam run results
│   ├── hparam_best_config.json  # Best hparam config
│   ├── exp1_full_iam_history.csv
│   ├── exp2_clean_iam_history.csv
│   ├── vlm_confusion_all.csv
│   └── training_curves.png
└── logs/
    ├── train_exp1_full_iam.log
    ├── train_exp2_clean_iam.log  # new run starts at line 304
    └── hparam_workers/          # per-run worker logs
```

---

## Results Summary (to be updated)

| Experiment | Train samples | Val samples | Best val CER | Test CER | Test WER |
|------------|--------------|-------------|-------------|----------|----------|
| Exp1: Full IAM | 6,482 | 976 | ~5.75% (ep85) | TBD | TBD |
| Exp2: Clean IAM | 4,338 | 616 | TBD (ep36: 9.58%) | TBD | TBD |
| Exp3a–f: +Synth | TBD | 976 | TBD | TBD | TBD |
| Exp4a–f: Clean+Synth | TBD | 616 | TBD | TBD | TBD |

*Beam search (w=50) and spell correction results to be added after full training.*
