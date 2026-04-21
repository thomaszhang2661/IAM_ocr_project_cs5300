# Handwritten Text Recognition on IAM
## VLM-Assisted Annotation Auditing, Synthetic Data Augmentation, and Language Model Rescoring for CRNN-CTC

**CS5300 Computer Vision — Final Project | Northeastern University | April 2026**
**Author:** Jian Thomas Zhang
**github:** https://github.com/thomaszhang2661/IAM_ocr_project_cs5300
---

## Overview

This project builds a CRNN-CTC handwritten text recognition (HTR) system on the IAM Handwriting Database, using it as a controlled platform to study **data-centric** improvements:

1. **Hyperparameter search** — 14-run parallel grid search → val CER 9.04% → 6.29%
2. **VLM annotation auditing** — Doubao Seed 2.0 Pro audits all 10,373 samples; prompt sensitivity analysis
3. **Synthetic data augmentation** — font-rendered rare-character oversampling, 1×–10× scale
4. **Beam search + KenLM LM** — 4-gram language model reduces CER from 7.53% → **6.41%**

**Best result: 6.41% test CER** (Exp3-10x + 4-gram KenLM, mixed-case vocabulary)

---

## Repository Structure

```
final_project/
├── htr_model/
│   ├── dataset.py          # LMDBDataset, collate_fn, Converter
│   ├── model.py            # CRNN v1 (LSTM)
│   └── model_v2.py         # CRNN v2 (BiGRU) — main model
├── data/
│   ├── iam_hf/             # IAM raw labels (train/val/test CSV)
│   ├── lmdb/               # LMDB image databases
│   │   ├── train/          # Full IAM training set (6,482 samples)
│   │   ├── train_clean/    # V1 VLM-cleaned training set (4,338 samples)
│   │   ├── train_clean_2/  # V2 VLM-cleaned training set (5,668 samples)
│   │   ├── train_synth/    # Synthetic augmentation pool (65,000 samples)
│   │   ├── val/
│   │   ├── val_clean/
│   │   └── val_clean_2/
│   └── generate_rare_texts.py
├── lm/
│   ├── build_lm.sh         # Downloads Gutenberg corpus, trains KenLM 4-gram
│   ├── word_4gram.arpa     # 150MB word 4-gram language model
│   └── char_6gram.arpa     # 35MB character 6-gram language model
├── train_iam.py            # Main training script
├── evaluate_iam.py         # Greedy CTC evaluation
├── decode_beam_v2.py       # Beam search with KenLM
├── analyze_char_errors.py  # Per-character CER via Levenshtein backtrace
├── analyze_confusion_matrix.py  # Model prediction confusion matrix
├── hparam_search.py        # Hyperparameter search orchestrator
├── vlm_inference/
│   ├── doubao_check.py     # VLM annotation audit (V1 prompt)
│   └── doubao_check_v2.py  # VLM annotation audit (V2 prompt)
├── results/
│   ├── paper/
│   │   ├── english_paper.md
│   │   └── chinese_paper.md
│   ├── slides.md           # Marp presentation source
│   ├── slides.html         # Rendered presentation (open in browser)
│   └── *.json              # All experiment results
├── checkpoints/            # Saved model checkpoints
└── logs/                   # Training and evaluation logs
```

---

## Setup

```bash
conda activate ocr_IAM
pip install -r requirements.txt

# For beam search language model
pip install pyctcdecode kenlm
```

**Requirements:** PyTorch ≥ 2.0, CUDA, lmdb, editdistance, pandas, tqdm

---

## Experiments

### Experiment Index

| ID | Description | Training Set | Synth | Best Test CER |
|---|---|---|---|---|
| Exp0 | Original (pre-hparam-search) | Full IAM, LSTM h=256 | 0× | 11.53% |
| Exp1 | Full IAM + optimal hparams | Full IAM (6,482) | 0× | 8.43% |
| Exp2 | V1 VLM-cleaned | V1 cleaned (4,338) | 0× | 10.33% |
| Exp3-1x | Full IAM + synth | Full IAM (6,482) | 1× | 7.93% |
| Exp3-3x | Full IAM + synth | Full IAM (6,482) | 3× | 7.66% |
| Exp3-5x | Full IAM + synth | Full IAM (6,482) | 5× | 7.63% |
| **Exp3-10x** | Full IAM + synth | Full IAM (6,482) | 10× | **7.52%** |
| Exp4-1x | V1 cleaned + synth | V1 cleaned (4,338) | 1× | 8.94% |
| Exp4-2x | V1 cleaned + synth | V1 cleaned (4,338) | 2× | 8.76% |
| Exp4-3x | V1 cleaned + synth | V1 cleaned (4,338) | 3× | 8.61% |
| Exp4-5x | V1 cleaned + synth | V1 cleaned (4,338) | 5× | 8.86% ↑ |
| Exp4-7x | V1 cleaned + synth | V1 cleaned (4,338) | 7× | 8.91% ↑ |
| Exp4-10x | V1 cleaned + synth | V1 cleaned (4,338) | 10× | 9.07% ↑ |
| Exp5-base | V2 VLM-cleaned | V2 cleaned (5,668) | 0× | 8.36% |
| Exp5-5x | V2 cleaned + synth | V2 cleaned (5,668) | 5× | 7.92% |
| Exp5-7x | V2 cleaned + synth | V2 cleaned (5,668) | 7× | 8.17% |
| Exp5-10x | V2 cleaned + synth | V2 cleaned (5,668) | 10× | 8.06% |

### Training

```bash
# Full IAM + 10× synthetic (best model)
python train_iam.py \
  --model v2 --run_name exp3_full_10x \
  --train_lmdb ./data/lmdb/train_full \
  --extra_train_lmdb ./data/lmdb/train_synth --synth_n 65000 \
  --val_lmdb ./data/lmdb/val_full \
  --epochs 150 --lr 3e-4 --hidden 512 \
  --batch_size 64 --weight_decay 1e-4 --dropout 0.1 --gpu 0
```

### Greedy Evaluation

```bash
python evaluate_iam.py \
  --checkpoint checkpoints/exp3_full_10x/best.pt \
  --split test --model v2 --hidden 512 --gpu 0
```

### Beam Search (KenLM)

```bash
# Build 4-gram LM first (one-time, ~15 min)
bash lm/build_lm.sh

# Decode with beam search
python decode_beam_v2.py \
  --checkpoint checkpoints/exp3_full_10x/best.pt \
  --split test \
  --lm_path lm/word_4gram.arpa \
  --confusion_min 999999 \
  --out_json results/beam_v3_exp3_full_10x.json \
  --gpu 0
```

---

## Key Results

### Beam Search Decoding (exp3-10x)

| Decoding | CER | Δ vs. greedy |
|---|---|---|
| CTC greedy | 7.53% | — |
| Beam + unigram vocab | 7.50% | −0.03% |
| Beam + 4-gram KenLM (lowercased vocab) | 6.64% | −0.89% |
| **Beam + 4-gram KenLM (mixed-case vocab)** | **6.41%** | **−1.12%** |

### VLM Annotation Auditing

| Prompt | Removed | Test CER | Effect |
|---|---|---|---|
| V1 (aggressive) | 33% | 10.33% | Harmful |
| V2 (conservative) | 12.6% | 8.36% | Neutral |
| No cleaning (baseline) | 0% | 8.43% | — |

### Novel Finding: Cross-Matrix Correlation

VLM annotation confusion pairs vs. model prediction confusion pairs:
**Spearman ρ = 0.559, p < 0.0001** (n=160 shared pairs)

Both systems struggle with the same character pairs (`o↔a`, `n↔m`, `r↔s`), confirming that these are genuine visual ambiguities in cursive handwriting — not annotation errors.

---

## Analysis Scripts

```bash
# Per-character error rate (Levenshtein backtrace)
python analyze_char_errors.py

# Model prediction confusion matrix
python analyze_confusion_matrix.py \
  --results_json results/test_results_exp3_full_10x.json
```

---

## Presentation

Slides are in `results/slides.md` (Marp format).

**View in browser:** Open `results/slides.html`

**Export to PDF** (requires Chrome/Chromium):
```bash
npx @marp-team/marp-cli results/slides.md --pdf --allow-local-files -o results/slides.pdf
```

**Export with VS Code:** Install the [Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode) extension, then use "Export Slide Deck → PDF".

---

## Papers

- `results/paper/ieee_paper.tex` — IEEE-format LaTeX source (main submission)
- `results/paper/english_paper.md` — Full English report (~12,000 words)
- `results/paper/chinese_paper.md` — Chinese version

---

## References

- Shi et al. (2016). An end-to-end trainable scene text recognition. *TPAMI*.
- Graves et al. (2006). Connectionist temporal classification. *ICML*.
- Li et al. (2021). TrOCR: Transformer-based OCR. *arXiv:2109.10282*.
- Retsinas et al. (2022). Best practices for a handwritten text recognition system. *DAS 2022*.
- Marti & Bunke (2002). The IAM database. *IJDAR*.
- Heafield (2011). KenLM: Faster and smaller language model queries. *WMT*.
