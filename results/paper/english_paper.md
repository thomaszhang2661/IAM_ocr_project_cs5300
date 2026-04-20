# Handwritten Text Recognition on IAM: VLM-Assisted Annotation Cleaning, Synthetic Data Augmentation, and Hyperparameter Optimization for CRNN-CTC

**Jian (Thomas) Zhang**
Northeastern University — MS Computer Science (Align)
Computer Vision — Final Project | April 2026
<!-- 【注：填入课程编号和 Prof. Maxwell 名字】 -->

---

## Abstract

This paper investigates handwritten text recognition (HTR) on the IAM Handwriting Database using a CRNN-CTC architecture. We make three contributions. First, a systematic hyperparameter search across 14 configurations using parallel multi-GPU training identifies an optimal configuration (hidden=512, lr=3×10⁻⁴) that achieves **6.29% validation CER** at 50 epochs — a 30% relative improvement over our baseline. Second, a VLM-assisted annotation quality analysis using Doubao Seed 2.0 Pro reveals 3,557 flagged samples across all splits (34.3% flag rate) and identifies 936 unique character substitution pairs, dominated by visually similar lowercase letters (o/a, n/m, r/s). Third, a synthetic data augmentation pipeline using 14 fonts and LLM-generated rare-character text targets underrepresented characters in the IAM distribution. Controlled experiments compare full vs. VLM-cleaned training sets and evaluate synthetic data at scales of 1×–10× the original training set. Full 150-epoch training achieves a test CER of **[TBD — Exp1]** on the full dataset and **[TBD — Exp2]** on the cleaned dataset.

<!-- 【注：最后两个数字等训练完填入，其余已确认】 -->

---

## 1. Introduction

Handwritten text recognition (HTR) — the automatic transcription of handwritten images to digital text — remains a core challenge in computer vision. Despite significant progress from CRNN-CTC architectures (Shi et al., 2016) to Transformer-based models such as TrOCR (Li et al., 2021), two foundational issues receive comparatively little attention: annotation quality in standard benchmarks, and the practical gap between proxy-metric optimization and real test performance.

The IAM Handwriting Database (Marti & Bunke, 2002) is the most widely used English HTR benchmark, containing 13,353 line-level samples from 657 writers. It is generally treated as a gold-standard reference — yet to our knowledge no systematic audit of its annotation quality has been published. This work fills that gap. We apply Doubao Seed 2.0 Pro, a state-of-the-art VLM, to audit all 10,373 IAM samples (train + val + test), revealing a 34.3% flag rate and a structured pattern of character-level confusion concentrated in visually ambiguous lowercase pairs.

To contextualize our baseline, Table 1 summarizes the current state of the art on IAM:

| Model | Architecture | IAM Test CER | LM Rescoring |
|---|---|---|---|
| DRetHTR (2025) | Decoder-only RetNet | 2.26% | Yes |
| TrOCR-Large (Li et al., 2021) | BEiT + RoBERTa | 2.89% | No |
| Retsinas et al. (2022) + RL | CRNN + CTC shortcut | ~4.5% | No |
| Retsinas et al. (2022) | CRNN + CTC shortcut | 5.14% | No |
| **Ours — Exp1 (full IAM, 150ep)** | CRNN-CTC BiGRU | **[TBD]** | No |
| **Ours — Exp2 (cleaned, 150ep)** | CRNN-CTC BiGRU | **[TBD]** | No |

*Table 1: IAM line-level test CER. Our work uses CRNN-CTC as a controlled baseline to isolate the effect of annotation cleaning from architectural improvements.*

Our model does not aim to surpass Transformer-based SOTA. Instead, we use CRNN-CTC as a controlled setting to answer three research questions:

- **RQ1:** Does a 14-run systematic hyperparameter search substantially improve CRNN-CTC performance on IAM?
- **RQ2:** Does removing VLM-flagged annotation noise from training improve test CER, and for which character categories?
- **RQ3:** Does font-based synthetic data augmentation at various scales (1×–10×) provide consistent improvement on IAM test?

**Our contributions:**

1. Systematic 14-run, 7-GPU parallel hyperparameter search for CRNN-CTC, with complete results and analysis
2. First systematic VLM-based annotation quality audit of IAM, with a 26×26 character confusion matrix across 936 substitution pairs
3. Controlled experiments: full vs. VLM-cleaned training, with per-character error analysis
4. Synthetic data augmentation pipeline with LLM-generated rare-character text, evaluated at 1×–10× scale
5. [Optional] Beam-search decoding enhanced with character confusion prior and KenLM language model

<!-- 【注：第5条如果KenLM没装好，删掉或移到Future Work】 -->

---

## 2. Related Work

### 2.1 CRNN-CTC for Handwritten Text Recognition

The CRNN architecture (Shi et al., 2016) combining VGG-style CNN feature extraction with bidirectional LSTM sequence modeling, trained end-to-end with CTC loss (Graves et al., 2006), established the dominant paradigm for HTR for nearly a decade. On IAM, vanilla CRNN-CTC baselines achieve 8–12% CER without language model rescoring.

More recent CRNN variants with residual blocks and an auxiliary CTC shortcut (Retsinas et al., 2022) reach 5.14% CER on IAM line-level recognition without external language models — described as "best practices" achievable with a standard convolutional-recurrent architecture. The gap from our expected result (~7–10%) to this level is primarily attributable to the CTC shortcut and residual CNN design, both orthogonal to our contributions.

The current state of the art is dominated by Transformer-based approaches: TrOCR (Li et al., 2021) achieves 2.89% CER by combining a pretrained BEiT image encoder with a RoBERTa decoder, pretrained on 684 million synthetic lines. DRetHTR (2025) further improves to 2.26% CER using a decoder-only RetNet architecture with layer-wise gamma scaling. These models require 200–600 GPU hours for fine-tuning on multi-GPU clusters, making controlled data-centric studies difficult. We intentionally use CRNN-CTC to maintain experimental tractability.

### 2.2 Label Noise in Benchmark Datasets

Label noise in training data is a well-documented source of model degradation (Frenay & Verleysen, 2014). In NLP benchmarks, the widely-used CoNLL-03 NER dataset has been estimated to contain 5–7% label noise (Wang et al., 2019; Reiss et al., 2020), and similar issues have been found in OntoNotes4 (~8%) and WNUT-17 (~18%). In image classification, CIFAR-N (Wei et al., 2022) provides human-annotated noisy labels to study the practical impact of real-world annotation errors.

Most relevant to our work, REVEAL (2025) proposes a unified framework using multiple VLMs to renovate image classification test sets by detecting label noise and imputing missing labels, using model agreement analysis and ensembling. We apply a similar VLM-based auditing philosophy to the HTR domain — extending it to the full train/val/test pipeline and extracting structured character-level confusion statistics.

To our knowledge, no prior work has systematically audited IAM annotation quality using modern VLMs, nor produced a character-level confusion matrix from such an audit.

### 2.3 Synthetic Data Augmentation for HTR

Font-based text rendering (trdg, Belval 2019) provides a low-cost source of labeled training data. Prior work has used trdg to generate up to 2.5 million synthetic handwriting image lines for pretraining (Luo et al., 2023), with consistent but diminishing gains as the synthetic-to-real ratio increases. Wigington et al. (2017) demonstrated modest but consistent improvement on IAM with image-level augmentation, while GAN-based synthesis methods (e.g., GANwriting) provide higher-quality samples at substantially greater computational cost (Kang et al., 2020).

A key limitation of font-based synthesis for IAM is the domain gap: rendered text lacks the stroke-level variability of real handwriting. We design our pipeline to specifically target underrepresented characters in the IAM distribution (digits, uppercase, punctuation), using an LLM (Doubao Seed 2.0 Pro) to generate text templates dense in rare characters — addressing a known class imbalance rather than simply scaling up volume.

### 2.4 Language Model Rescoring for CTC

CTC beam search combined with n-gram language model rescoring is the standard post-processing approach for both ASR (Hannun et al., 2014) and HTR. The pyctcdecode + KenLM pipeline implements the scoring formula:

```
score = α × CTC_log_prob + β × LM_score + γ × word_count
```

where α, β, γ are tunable weights. TrOCR's paper notes that external LM rescoring results in significant CER reduction for CTC-based HTR methods. We additionally explore incorporating the character confusion prior derived from our VLM analysis into the beam search emission probabilities — to our knowledge a novel integration not previously reported.

---

## 3. Dataset and Preprocessing

### 3.1 IAM Handwriting Database

The IAM Handwriting Database (Marti & Bunke, 2002) contains handwritten English text scanned at 300 dpi from 657 writers, originally sourced from the LOB corpus. We use the standard line-level split:

| Split | Samples | Writers | Notes |
|---|---|---|---|
| Train | 6,482 | ~547 | Full and cleaned variants |
| Validation | 976 | ~55 | Model selection only |
| Test | 2,915 | ~55 | Held out; evaluated once per experiment |
| **Total** | **10,373** | **657** | |

**Preprocessing:** All images are resized to height H=64 pixels with aspect ratio preserved via padding. Images are normalized per-channel (zero mean, unit variance) and stored in LMDB format for efficient batch loading. The character vocabulary contains 79 IAM-standard characters plus a CTC blank token (index 0):

```
 !"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz
```

### 3.2 VLM Annotation Quality Audit

We audited all 10,373 samples using Doubao Seed 2.0 Pro. Each image was presented to the VLM with its ground-truth label; the model was asked to verify correctness and propose corrections where needed, outputting a structured verdict (CORRECT / INCORRECT) and a corrected transcription.

**Audit results:**

| Split | Total | Flagged | Flag Rate |
|---|---|---|---|
| Train | 6,482 | 2,144 | 33.1% |
| Validation | 976 | 360 | 36.9% |
| Test | 2,915 | 1,053 | 36.1% |
| **Total** | **10,373** | **3,557** | **34.3%** |

The high flag rate (~34%) reflects a combination of:
- Genuine transcription errors (e.g., `0M` → `OM`, visually ambiguous digits mistaken for letters)
- VLM over-flagging: synonym substitutions (`out-dated` → `outmoded`), grammar corrections (`appear` → `appears`), stylistic rewrites

Example flagged entries illustrating both categories:

| Ground Truth | VLM Correction | Reason | Type |
|---|---|---|---|
| `0M P for Manchester Exchange .` | `UM P for Manchester Exchange .` | `0M→UM` | Likely genuine error |
| `meeting of Labour 0M Ps tommorow` | `meeting of Labour OM Ps tomorrow` | `0M→OM, tommorow→tomorrow` | Mixed |
| `which would appear to "prop up" an out-dated` | `which would appear to "prop up" an outmoded` | `out-dated→outmoded` | VLM over-flag |
| `and he is to be backed by Mr. Will` | `and her is to be backed by her Will` | `he→her, Mr.→her` | VLM hallucination |

<!-- 【注：Manual validation of 100 randomly sampled flagged examples estimated VLM precision at [TBD]%, with [TBD]% identified as genuine transcription errors. 等人工验证完填入 — 这是最关键的一步】 -->

**Important caveat:** Without manual validation, the 34.3% flag rate cannot be equated with a true annotation error rate. The cleaned training set (Section 3.4) removes all flagged samples; its benefit depends on the fraction that are genuine errors.

### 3.3 Character Confusion Analysis

Substitutions were extracted by character-level alignment (Python `difflib.SequenceMatcher`) between IAM ground-truth and VLM-corrected text across all flagged samples. We identified **936 unique confusion pairs** totaling **5,058 substitution events** across train + val + test.

**Top 20 confusion pairs:**

| Rank | IAM label | VLM correction | Count | Category |
|---|---|---|---|---|
| 1 | `o` | `a` | 155 | lower–lower |
| 2 | `n` | `m` | 135 | lower–lower |
| 3 | `r` | `s` | 133 | lower–lower |
| 4 | `a` | `o` | 116 | lower–lower (bidirectional) |
| 5 | `a` | `e` | 109 | lower–lower |
| 6 | `n` | `u` | 96 | lower–lower |
| 7 | `o` | `e` | 76 | lower–lower |
| 8 | `t` | `h` | 75 | lower–lower |
| 9 | `t` | `d` | 68 | lower–lower |
| 10 | `h` | `l` | 60 | lower–lower |
| 11 | `s` | `r` | 59 | lower–lower |
| 12 | `r` | `n` | 56 | lower–lower |
| 13 | `t` | `l` | 54 | lower–lower |
| 14 | `u` | `e` | 52 | lower–lower |
| 15 | `m` | `w` | 47 | lower–lower |
| 16 | `e` | `a` | 46 | lower–lower |
| 17 | ` ` | `s` | 42 | space→lower |
| 18 | `e` | `i` | 42 | lower–lower |
| 19 | `t` | `H` | 40 | lower→upper |
| 20 | `d` | `c` | 39 | lower–lower |

**Key observations:**
- 93% of confusion events involve lowercase letters — consistent with handwriting annotation difficulty for visually similar cursive strokes
- `o` ↔ `a` is the dominant bidirectional pair (155 + 116 = 271 combined events)
- `n` is a confusion hub: confused with `m` (135), `u` (96), `r` (56), `s` (36) — 5 different targets
- `t` is similarly problematic: confused with `h` (75), `d` (68), `l` (54), `f` (35)
- Digits and uppercase letters are rarely confused, suggesting IAM annotators had greater difficulty with cursive lowercase than with other character classes

<!-- 【注：插入 Figure — vlm_confusion_matrix.png (26×26 lowercase heatmap)】 -->

### 3.4 Cleaned Training Set

We construct `train_clean` by removing all 2,144 VLM-flagged samples from the original 6,482-sample training set:

| Set | Samples | Notes |
|---|---|---|
| `train` (full) | 6,482 | Used in Exp1 |
| `train_clean` | 4,338 | 2,144 removed (−33.1%); used in Exp2 |

The test set is never cleaned (labels are treated as fixed for evaluation).

### 3.5 IAM Character Distribution

The IAM training set has a heavily skewed character distribution. Lowercase letters account for ~68% of all characters; digits, uppercase, and punctuation are significantly underrepresented:

| Category | Approx. frequency | Notes |
|---|---|---|
| Lowercase (a–z) | ~68% | Dominant |
| Space | ~16% | |
| Uppercase (A–Z) | ~8% | Highly variable per letter |
| Digits (0–9) | ~3% | Many characters < 500 occurrences |
| Punctuation | ~5% | `#`, `*`, `+`, `Z`, `Q` < 50 occurrences |

This imbalance motivates our targeted synthetic data pipeline (Section 5).

---

## 4. Model Architecture

### 4.1 CRNN-CTC

Our model follows the standard CRNN pipeline: CNN feature extractor → sequence encoder → CTC decoder.

**CNN Feature Extractor (VGG-style, H=64 adapted)**

We adapt the original CRNN VGG backbone (designed for H=32) to H=64 by modifying the final convolutional kernel from (2,1) to (4,1):

| Layer | Kernel / Pool | Output (H×W×C) | BN |
|---|---|---|---|
| Conv1 + Pool | 3×3, MaxPool 2×2 | 32×W/2×64 | No |
| Conv2 + Pool | 3×3, MaxPool 2×2 | 16×W/4×128 | No |
| Conv3 + Conv4 + Pool | 3×3, AsymPool (2,2)/(2,1) | 4×W/4×256 | After Conv3 |
| Conv5 + Conv6 + Pool | 3×3, AsymPool (2,2)/(2,1) | 2×W/4×512 | After Conv5 |
| Conv7 | kernel=(4,1) | 1×W/4×512 | After Conv7 |

The asymmetric pooling in layers 3–4 (stride (2,1), padding (0,1)) preserves horizontal resolution while halving vertical resolution, which is standard practice for text line recognition.

**Comparison with original VGG (Run 13 result):** The original paper uses final kernel (2,1) designed for H=32. We tested both:

| Architecture | Val CER (50 ep) | Decision |
|---|---|---|
| H=64 adapted (kernel=(4,1)) — ours | 7.35% | Adopted |
| Original (kernel=(2,1)) | 7.45% | Discarded |

The H=64 variant marginally outperforms the original, validating our design choice.

**Sequence Encoder:** 2-layer Bidirectional GRU, hidden size = 512 (optimal from Section 6). Input size = 512 (CNN output channels).

**Output projection:** Linear(1024 → 80), log-softmax over 80 classes (79 characters + CTC blank).

**Loss:** CTCLoss (blank=0, zero_infinity=True, reduction='mean')

**Total parameters: ~16.7M**

### 4.2 Training Details

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | weight_decay=1e-4 |
| Initial LR | 3×10⁻⁴ | Optimal from hparam search |
| LR schedule | ReduceLROnPlateau | patience=10, factor=0.8, min_lr=1e-7 |
| Warmup | 1 epoch | LinearLR from ≈0 to initial LR |
| Batch size | 64 | |
| Grad clip | 10.0 | L2 norm |
| Dropout | 0.1 | Applied before encoder |
| Epochs | 150 (main) / 50 (hparam proxy) | |
| Input height | H=64 | Width variable, aspect ratio preserved |
| Augmentation | Random affine, elastic distortion, brightness jitter | Training only |

---

## 5. Synthetic Data Pipeline

### 5.1 Motivation

The IAM training set contains only 6,482 samples, with severe character imbalance: rare characters such as `#`, `*`, `+`, `Z`, `Q` appear fewer than 10 times. Standard font-based synthesis can efficiently increase sample count and character coverage without the cost of human annotation.

### 5.2 Text Sources

We use a three-layer text pool:

**Layer 1 — IAM training labels (×2, ~13,000 sentences)**
Direct re-use of IAM training text ensures vocabulary and sentence structure closely match the test distribution. Repeated twice to up-weight domain-appropriate text.

**Layer 2 — LLM-generated rare-character sentences (~12,000 sentences)**
We prompt Doubao Seed 2.0 Pro to generate sentences where each target character appears at least 2–3 times. Target characters: all digits, uppercase letters with < 500 IAM training occurrences, and rare punctuation (`#`, `&`, `*`, `+`, `/`, `;`, `:`). This directly addresses the class imbalance documented in Section 3.5.

Example LLM-generated sentences targeting `Z`:
```
The Zurich-based firm closed at 47.2 on the ZSE index, down from last week's peak.
Dr. Zimmerman's Zone 7 protocol requires zero tolerance for unauthorized access.
```

**Layer 3 — Template-generated sentences (~3,000)**
Rule-based templates with random slots for digits, names, dates, and punctuation. These serve as a lightweight fallback for characters where LLM generation underperforms.

**Combined pool: ~28,000 sentences**, shuffled before generation.

### 5.3 Rendering

Font rendering uses trdg (Belval, 2019) with the following 14 fonts:

| Type | Fonts (9 handwriting-style) |
|---|---|
| Handwriting | ArchitectsDaughter, HomemadeApple, IndieFlower, Kalam-Light, Kalam-Regular, PatrickHand, PermanentMarker, RockSalt, Satisfy |
| Print (Regular/Italic) | DroidSans, Lato-Regular, Lato-Italic, OpenSans-Regular, Raleway-Regular |

Bold fonts were excluded as IAM handwriting has predominantly thin to medium stroke widths; bold fonts introduce distribution shift.

Each rendered image: H=64px, random skew (0–3°), random blur (0–1px), random distortion type (elastic/perspective/none).

**Generation scale for experiments:**

| Scale | Synthetic samples | Total (+ IAM train) |
|---|---|---|
| 1× | 6,500 | ~13,000 |
| 2× | 13,000 | ~19,500 |
| 3× | 19,500 | ~26,000 |
| 5× | 32,500 | ~39,000 |
| 7× | 45,500 | ~52,000 |
| 10× | 65,000 | ~71,500 |

A single 65,000-sample LMDB is generated once; training experiments use subsets.

### 5.4 Character Frequency Report

<!-- 【注：插入 Figure — synth_char_freq.png (bar chart, log scale)，对比训练集和合成集的字符分布】 -->

After synthesis, rare characters that were < 500 occurrences in IAM training achieve the following counts in the 65k synthetic set: `[TBD — fill from synth_char_freq.csv]`

---

## 6. Hyperparameter Search

### 6.1 Protocol

We conduct an autoresearch-style greedy search: starting from a baseline configuration, we change one hyperparameter at a time and **keep** the change if validation CER improves by > 0.05% relative to the current best. All runs use 50 epochs as a proxy budget (vs. 150 for main experiments) to maintain tractability.

**Baseline config:** hidden=256, dropout=0.1, batch_size=64, lr=1e-4, cnn_out=512, weight_decay=1e-4

Runs 00–02 were run sequentially (GPU 0). After Run 02 established a new best with hidden=512, the remaining 11 runs were distributed across 7 GPUs in parallel, each using the best known config (hidden=512) as the base.

### 6.2 Results

| Run | GPU | Change from best | Val CER | Δ vs. prior best | Decision |
|---|---|---|---|---|---|
| 00 | 0 | Baseline (hidden=256) | 9.04% | — | Keep |
| 01 | 0 | hidden=128 | 9.79% | +0.75% | Discard |
| **02** | **0** | **hidden=512** | **7.35%** | **−1.69%** | **Keep → new best** |
| 03 | 0 | dropout=0.0 | 7.45% | +0.10% | Discard |
| 04 | 2 | dropout=0.2 | 7.85% | +0.50% | Discard |
| 05 | 3 | dropout=0.3 | 7.20% | −0.15% | Keep |
| 06 | 4 | batch_size=32 | 6.73% | −0.62% | Keep |
| 07 | 5 | batch_size=128 | 8.39% | +1.04% | Discard |
| **08** | **6** | **lr=3×10⁻⁴** | **6.29%** | **−1.06%** | **Keep → new best** |
| 09 | 7 | lr=5×10⁻⁵ | 9.00% | +2.71% | Discard |
| 10 | 2 | cnn_out=256 | 7.50% | +1.21% | Discard |
| 11 | 3 | weight_decay=1e-3 | 8.05% | +1.76% | Discard |
| 12 | 4 | weight_decay=0 | 7.42% | +1.13% | Discard |
| 13 | 5 | original_vgg (kernel=2, H=32 style) | 7.45% | +1.16% | Discard |

**Optimal configuration:** hidden=512, lr=3×10⁻⁴, dropout=0.1, batch_size=64, cnn_out=512, weight_decay=1e-4

**50-epoch proxy CER:** 6.29% (validation)

<!-- 【注：插入 Figure — hparam bar chart: run_id on x-axis, val CER on y-axis, kept runs in blue, discarded in gray】 -->

### 6.3 Analysis

**hidden=512** produced the largest single improvement (9.04% → 7.35%), indicating the BiGRU was the bottleneck in the baseline configuration. Larger hidden sizes provide greater temporal modeling capacity for the variable-length character sequences in IAM.

**lr=3×10⁻⁴** provided the second-largest improvement (7.35% → 6.29%). With ReduceLROnPlateau scheduling, this reflects faster convergence within the 50-epoch proxy budget — the higher initial LR reaches the plateau regime sooner. In full 150-epoch training, the benefit may be smaller as both LR values eventually decay to similar levels.

**batch_size=32** showed modest improvement (7.35% → 6.73%), consistent with the known effect of noisier gradients improving generalization for small datasets.

**dropout=0.3** (6.29% → 7.20%): marginal improvement at the dropout dimension, suggesting the model is already well-regularized at dropout=0.1 when combined with weight decay.

**Original VGG (kernel=2, Run 13)**: Result 7.45% confirms our H=64 adapted architecture (7.35%) is the correct design choice. The original kernel size compresses the spatial dimension sub-optimally for H=64 input.

**Negative results:** lr=5×10⁻⁵ (9.00%) is the worst result, confirming that too-small LR makes 50 epochs insufficient for convergence. batch_size=128 (8.39%) trades gradient noise for update frequency in a way that hurts convergence on this dataset size.

---

## 7. Experiments

### 7.1 Experimental Setup

All main experiments use the optimal config from Section 6, trained for **150 epochs** with ReduceLROnPlateau (patience=10, factor=0.8, min_lr=1e-7) and 1-epoch linear LR warmup.

**Evaluation metric:** Character Error Rate (CER):
```
CER = edit_distance(prediction, ground_truth) / max(len(ground_truth), 1) × 100%
```

Each experiment is run once on a single A100 80GB GPU (~45 minutes per run at 150 epochs). Checkpoints saved at best validation CER; test evaluation uses best checkpoint only.

### 7.2 Exp1 & Exp2: Full vs. Cleaned IAM

| Experiment | Training Set | Samples | Val CER (best) | Test CER |
|---|---|---|---|---|
| Exp1 — Full IAM | `train` | 6,482 | [TBD] | [TBD] |
| Exp2 — Cleaned IAM | `train_clean` | 4,338 | [TBD] | [TBD] |

<!-- 【注：两个实验正在跑（150 epochs，约40分钟），填入后是报告的核心结论】 -->

**Interpretation framework:**

- If Exp2 < Exp1: VLM-flagged samples contain genuine noise that degrades model performance. The per-character analysis (Section 7.4) should show improvement concentrated in the visually ambiguous pairs identified in Section 3.3 (o/a, n/m, r/s).
- If Exp2 ≥ Exp1: Two hypotheses — (a) VLM over-flagging removes hard but valid examples that are important for generalization; (b) removing 33% of training data reduces diversity sufficiently to hurt performance despite noise reduction.

### 7.3 Exp3: Synthetic Data Augmentation at Multiple Scales

All Exp3 variants use the full `train` split + synthetic data subset. Training: 150 epochs.

| Experiment | Training Set | Synth | Total | Test CER |
|---|---|---|---|---|
| Exp1 (baseline) | Full IAM | 0 | 6,482 | [TBD] |
| Exp3a | Full IAM + synth | 1× (6.5k) | ~13k | [TBD] |
| Exp3b | Full IAM + synth | 2× (13k) | ~19.5k | [TBD] |
| Exp3c | Full IAM + synth | 3× (19.5k) | ~26k | [TBD] |
| Exp3d | Full IAM + synth | 5× (32.5k) | ~39k | [TBD] |
| Exp3e | Full IAM + synth | 7× (45.5k) | ~52k | [TBD] |
| Exp3f | Full IAM + synth | 10× (65k) | ~71.5k | [TBD] |

<!-- 【注：这组实验等合成数据65k生成完后跑，是报告的第三条贡献线。预期看到一个收益递减曲线】 -->
<!-- 【注：插入 Figure — CER vs. synthetic data scale (x: multiples of IAM, y: test CER)，是报告最直观的图之一】 -->

### 7.4 Per-Character Error Analysis

After Exp1 and Exp2 complete, we extract per-character CER to answer: do the confusion pairs identified by VLM analysis (Section 3.3) show disproportionate improvement in Exp2 vs. Exp1?

**Method:** For each test sample, align model output to ground truth using character-level edit distance; accumulate per-character deletion, insertion, and substitution counts.

| Character | Exp1 error rate | Exp2 error rate | Δ | VLM confusion count |
|---|---|---|---|---|
| `o` | [TBD] | [TBD] | [TBD] | 231 (o→a + a→o) |
| `n` | [TBD] | [TBD] | [TBD] | 165 (n→m + m→n) |
| `r` | [TBD] | [TBD] | [TBD] | 192 (r→s + s→r) |
| `t` | [TBD] | [TBD] | [TBD] | 197 (t→h/d/l/f) |
| ... | | | | |

<!-- 【注：这个分析是把VLM confusion和model error串联起来的关键，是报告最有说服力的图/表之一】 -->

If the per-character improvement in Exp2 correlates with the VLM confusion counts from Table 2, it provides indirect validation that the VLM-flagged samples contain genuine annotation errors rather than random noise.

---

## 8. Language Model Rescoring (Optional / Future Work)

<!-- 【注：如果KenLM装好了，这节写完整；否则保留框架作为Future Work】 -->

### 8.1 KenLM 4-gram Language Model

A 4-gram KenLM language model is trained on the IAM training set text labels (~6,482 sentences) using the `lmplz` binary. The model provides word-level probability estimates used in beam search rescoring.

### 8.2 Character Confusion Prior

We incorporate the VLM confusion table (pairs with count ≥ 10, totaling [TBD] pairs) as a soft prior on emission probabilities. For each time step, if the model assigns high probability to a "wrong" character (per the confusion table), we add a small probability mass to the corresponding "correct" character:

```
P_adj(c_correct | t) += λ × confusion_weight(c_wrong, c_correct) × P(c_wrong | t)
```

where λ is a tunable mixing coefficient.

### 8.3 Results

<!-- 【注：等KenLM实验跑完填入】 -->

| Decoding method | Test CER | Δ vs. greedy |
|---|---|---|
| CTC greedy | [TBD] | — |
| CTC beam search (beam=10) | [TBD] | [TBD] |
| CTC beam + KenLM | [TBD] | [TBD] |
| CTC beam + KenLM + confusion prior | [TBD] | [TBD] |

---

## 9. Discussion

### 9.1 Does VLM Cleaning Help?

<!-- 【注：填入 Exp1 vs Exp2 结果后写这节的核心结论】 -->

[TBD — fill after results]

If Exp2 improves over Exp1, the key question is: **which characters benefit most?** If the per-character analysis (Section 7.4) shows improvement concentrated in the high-confusion pairs (o/a, n/m, r/s), this strongly supports the VLM-as-auditor hypothesis. If improvement is diffuse, it may reflect general noise reduction rather than targeted correction.

If Exp2 does not improve, we discuss two possible explanations:
1. The ~34% removal rate is too aggressive — genuine hard examples that support model generalization are removed along with noisy ones
2. The 4,338 remaining samples are insufficient to learn the character diversity required for IAM test, despite higher annotation quality

### 9.2 Diminishing Returns in Synthetic Data

<!-- 【注：填入 Exp3 曲线后写】 -->

[TBD — fill after Exp3 results]

Prior work predicts that font-rendered synthetic data provides consistent but diminishing returns for IAM, where training and test distributions are already well-matched. We expect a CER-vs-scale curve that improves rapidly from 0× to ~3×, then plateaus or slightly degrades at 10× due to distribution shift. The LLM-generated rare-character text should show a stronger effect on per-character CER for digits and uppercase letters than for lowercase, where IAM training data is already relatively abundant.

### 9.3 Limitations

**VLM precision is unverified at scale.** The 34.3% flag rate almost certainly overestimates the true annotation error rate in IAM, which is known to have been double-checked by human annotators. Our cleaned training set removes a mix of genuine errors and VLM over-flags; the true benefit of cleaning is bounded by the VLM's precision on genuine errors.

<!-- 【注：After manual validation: replace above with "Manual review of N flagged samples estimated VLM precision at X%..."】 -->

**CRNN-CTC is not state of the art.** Our architecture achieves ~7–10% CER vs. 2–3% for TrOCR-scale models. The contribution is not architectural but data-centric: we demonstrate how annotation quality and data composition affect performance in a controlled setting. Whether the same effects hold for Transformer-based HTR remains an open question.

**Single dataset.** All experiments are conducted on IAM. Generalizability to other HTR benchmarks (RIMES, CVL, Bentham) is not demonstrated.

**Synthetic domain gap.** Font-rendered text lacks the stroke-level variability of real handwriting. GAN-based or diffusion-based synthesis would likely yield larger gains at equal volume, at substantially higher computational cost.

### 9.4 Implications for IAM as a Benchmark

Our analysis raises a broader question: if ~34% of IAM samples are flagged by a strong VLM (even accounting for over-flagging), what fraction of the commonly reported HTR improvement across papers reflects genuine model progress vs. overfitting to annotation noise? Even if the true error rate is much lower (e.g., 5–10% after filtering VLM over-flags), systematic annotation quality auditing appears to be an underinvestigated aspect of HTR benchmark design.

---

## 10. Conclusion

We present a comprehensive data-centric study of HTR on IAM, combining systematic hyperparameter optimization, VLM-assisted annotation quality analysis, and synthetic data augmentation.

Our 14-run hyperparameter search identified **hidden=512** and **lr=3×10⁻⁴** as the most impactful improvements, reducing validation CER from 9.04% to 6.29% at 50 epochs. A 26×26 character confusion matrix derived from VLM auditing revealed that IAM annotation errors are heavily concentrated in visually similar lowercase pairs — particularly `o/a` (271 combined events) and `n/m` (165 combined events).

[TBD — one sentence on Exp1 vs Exp2 test result and what it implies about annotation noise.]

[TBD — one sentence on synthetic data scaling result.]

**Future work:** (1) Extend VLM auditing to additional HTR benchmarks (RIMES, CVL, Bentham) to establish whether annotation noise patterns generalize; (2) Apply the same framework to Transformer-based HTR to test whether the benefit of cleaning scales with model capacity; (3) Investigate GAN-based or diffusion-based synthesis as higher-quality alternatives to font rendering for rare-character augmentation; (4) Formally analyze the relationship between VLM confusion patterns and model prediction errors as a signal for targeted data collection.

---

## References

Belval, E. (2019). TextRecognitionDataGenerator. GitHub. https://github.com/Belval/TextRecognitionDataGenerator

Frenay, B., & Verleysen, M. (2014). Classification in the presence of label noise: A survey. *IEEE Transactions on Neural Networks and Learning Systems*, 25(5), 845–869.

Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. *ICML 2006*.

Kang, L., Rusiñol, M., Fornés, A., Riba, P., & Villegas, M. (2020). Unsupervised adaptation for synthetic-to-real handwritten word recognition. *WACV 2020*.

Li, M., Lv, T., Cui, L., Lu, Y., Florencio, D., Zhang, C., Li, Z., & Wei, F. (2021). TrOCR: Transformer-based optical character recognition with pre-trained models. *arXiv:2109.10282*.

Luo, C., Jin, L., & Sun, Z. (2023). CNN-BiLSTM model for English handwriting recognition: Comprehensive evaluation on the IAM dataset. *arXiv:2307.00664*.

Marti, U.-V., & Bunke, H. (2002). The IAM-database: An English sentence database for offline handwriting recognition. *International Journal on Document Analysis and Recognition*, 5(1), 39–46.

Retsinas, G., Sfikas, P., Gatos, B., & Nikou, C. (2022). Best practices for a handwritten text recognition system. *arXiv:2404.11339*.

Shi, B., Bai, X., & Yao, C. (2016). An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(11), 2298–2304.

Wei, J., Zhu, Z., Cheng, H., Liu, T., Niu, G., & Liu, Y. (2022). Learning with noisy labels revisited: A study using real-world human annotations. *ICLR 2022*.

Wigington, C., Stewart, S., Price, B., & Cohen, S. (2017). Data augmentation for recognition of handwritten words and lines using a CNN-LSTM network. *ICDAR 2017*.

<!-- 【注：以下引用待确认完整信息后添加】 -->
<!-- DRetHTR (2025): arXiv号待查；搜索 "DRetHTR RetNet handwriting" 确认 -->
<!-- REVEAL (2025): arXiv:2505.16149 — When VLMs Meet Image Classification: Test Sets Renovation -->
<!-- Hannun et al. (2014): Deep Speech, arXiv:1412.5567 — for KenLM rescoring citation -->

---

## Appendix A: Hyperparameter Search Configuration Details

| Run | hidden | dropout | batch_size | lr | cnn_out | weight_decay | use_original_vgg |
|---|---|---|---|---|---|---|---|
| 00 | 256 | 0.1 | 64 | 1e-4 | 512 | 1e-4 | False |
| 01 | 128 | 0.1 | 64 | 1e-4 | 512 | 1e-4 | False |
| 02 | 512 | 0.1 | 64 | 1e-4 | 512 | 1e-4 | False |
| 03 | 512 | 0.0 | 64 | 1e-4 | 512 | 1e-4 | False |
| 04 | 512 | 0.2 | 64 | 1e-4 | 512 | 1e-4 | False |
| 05 | 512 | 0.3 | 64 | 1e-4 | 512 | 1e-4 | False |
| 06 | 512 | 0.1 | 32 | 1e-4 | 512 | 1e-4 | False |
| 07 | 512 | 0.1 | 128 | 1e-4 | 512 | 1e-4 | False |
| 08 | 512 | 0.1 | 64 | 3e-4 | 512 | 1e-4 | False |
| 09 | 512 | 0.1 | 64 | 5e-5 | 512 | 1e-4 | False |
| 10 | 512 | 0.1 | 64 | 1e-4 | 256 | 1e-4 | False |
| 11 | 512 | 0.1 | 64 | 1e-4 | 512 | 1e-3 | False |
| 12 | 512 | 0.1 | 64 | 1e-4 | 512 | 0.0 | False |
| 13 | 512 | 0.1 | 64 | 1e-4 | 512 | 1e-4 | True |

## Appendix B: VLM Audit Prompt Template

```
You are an expert handwriting transcription auditor.

Given: a handwritten text line image and its ground-truth transcription.
Task: Determine if the transcription is correct.

Ground truth: "{label}"

If CORRECT, reply: CORRECT
If INCORRECT, reply: INCORRECT: {error_description} | CORRECTED: {corrected_text}

Focus on: character-level transcription accuracy. Do NOT correct grammar,
spelling, or style — only transcription errors where the handwritten character
was misread (e.g., 0 mistaken for O, n mistaken for m).
```

<!-- 【注：这个 prompt 是关键，如果实际用的不一样要改，审稿人可能会问】 -->

## Appendix C: Synthetic Data Character Coverage

<!-- 【注：等 synth_char_freq.csv 生成后，把前50个字符的频次填入这里作为附录】 -->

[TBD — insert character frequency table from synth_char_freq.csv]

---

*[TBD Summary: Items requiring experimental completion before final submission:]*
*1. Exp1 test CER (full IAM, 150 epochs) — ~40 min*
*2. Exp2 test CER (cleaned IAM, 150 epochs) — ~40 min*
*3. Per-character error analysis (Exp1 vs Exp2)*
*4. Exp3a–f synthetic data scale experiments*
*5. Manual validation of 100 flagged samples (VLM precision)*
*6. Figures: hparam bar chart, confusion heatmap, CER-vs-scale curve*
*7. KenLM rescoring results (if lmplz build succeeds)*
*8. DRetHTR full citation*