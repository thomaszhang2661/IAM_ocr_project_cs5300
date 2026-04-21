# Handwritten Text Recognition on IAM: VLM-Assisted Annotation Auditing, Synthetic Data Augmentation, and Language Model Rescoring for CRNN-CTC

**Jian (Thomas) Zhang**
Northeastern University — MS Computer Science (Align)
CS5300 Computer Vision — Final Project | April 2026

---

## Abstract

This paper presents a data-centric study of handwritten text recognition (HTR) on the IAM Handwriting Database using a CRNN-CTC architecture. We make four contributions. First, a systematic 14-run hyperparameter search identifies an optimal configuration (BiGRU hidden=512, lr=3×10⁻⁴) that achieves **6.29% validation CER** at 50 epochs — a 30% relative improvement over a vanilla baseline. Second, a VLM-assisted annotation quality audit using Doubao Seed 2.0 Pro reveals structured character-level confusion patterns, and demonstrates that prompt engineering critically determines cleaning outcome: an aggressive prompt removing 33% of training data degrades test CER from 8.43% to 10.33%, while a conservative prompt removing 12.6% has negligible effect. A novel cross-matrix analysis shows that the VLM's annotation confusion pairs and the model's prediction confusion pairs are significantly correlated (Spearman ρ=0.559, p<0.0001 across 160 shared pairs), indicating both reflect the same underlying visual ambiguity in handwritten cursive strokes rather than system-specific biases. Third, a synthetic data augmentation pipeline targeting underrepresented characters achieves **7.52% test CER** at 10× scale — an 11% relative improvement — with per-character analysis confirming the largest gains on rare symbols (`#`, `;`, `:`, uppercase). Fourth, beam search with a KenLM 4-gram language model trained on public English corpora reduces CER to **6.41%**, a 1.12% absolute improvement over greedy decoding, with a key insight that vocabulary case preservation is critical for proper-noun-heavy datasets.

---

## 1. Introduction

Handwritten text recognition (HTR) — the automatic transcription of handwritten images to digital text — remains a core challenge in computer vision. Despite significant progress from CRNN-CTC architectures (Shi et al., 2016) to Transformer-based models such as TrOCR (Li et al., 2021), two foundational issues receive comparatively little attention: annotation quality in standard benchmarks, and the practical gap between training data composition and model robustness.

The IAM Handwriting Database (Marti & Bunke, 2002) is the most widely used English HTR benchmark, containing 13,353 line-level samples from 657 writers. It is generally treated as a gold-standard reference. This work applies Doubao Seed 2.0 Pro, a state-of-the-art VLM, to audit all 10,373 IAM samples (train + val + test), revealing structured annotation error patterns and — crucially — demonstrating that the design of the VLM auditing prompt determines whether cleaning helps or hurts downstream performance.

To contextualize our results, Table 1 summarizes the current state of the art on IAM:

| Model | Architecture | IAM Test CER | LM |
|---|---|---|---|
| DRetHTR (2025) | Decoder-only RetNet | 2.26% | Yes |
| TrOCR-Large (Li et al., 2021) | BEiT + RoBERTa | 2.89% | No |
| Retsinas et al. (2022) | CRNN + CTC shortcut | 5.14% | No |
| **Ours — Exp3 (full + 10× synth) + beam** | CRNN-CTC BiGRU | **6.41%** | 4-gram |
| **Ours — Exp3 (full + 10× synth), greedy** | CRNN-CTC BiGRU | **7.52%** | No |
| **Ours — Exp1 (full IAM), greedy** | CRNN-CTC BiGRU | **8.43%** | No |

*Table 1: IAM line-level test CER. CRNN-CTC is used as a controlled setting to isolate data-centric effects from architectural improvements.*

Our model does not aim to surpass Transformer-based SOTA. Instead, we use CRNN-CTC as a controlled setting to answer four research questions:

- **RQ1:** Does systematic hyperparameter search substantially improve CRNN-CTC performance on IAM?
- **RQ2:** Does VLM-assisted annotation cleaning improve test CER, and how sensitive is the result to the cleaning aggressiveness?
- **RQ3:** Does font-based synthetic data augmentation at various scales provide consistent improvement, particularly for rare characters?
- **RQ4:** Does word-level n-gram language model rescoring provide meaningful gains over unigram-based beam search?

**Our contributions:**

1. Systematic 14-run, 7-GPU parallel hyperparameter search for CRNN-CTC, reducing val CER from 9.04% to 6.29%
2. First systematic VLM-based annotation quality audit of IAM, with a 26×26 character confusion matrix and a controlled experiment showing that cleaning aggressiveness critically mediates whether VLM-based cleaning helps or hurts
3. Synthetic data pipeline targeting underrepresented characters, evaluated at 1×–10× scale with per-character error analysis confirming targeted rare-character improvement
4. Beam search decoding with KenLM 4-gram LM trained on public corpora, achieving 6.41% test CER (best in our study); vocabulary case preservation discovered as a critical factor for proper-noun-heavy benchmarks

---

## 2. Related Work

### 2.1 CRNN-CTC for Handwritten Text Recognition

The CRNN architecture (Shi et al., 2016) combining VGG-style CNN feature extraction with bidirectional LSTM sequence modeling, trained end-to-end with CTC loss (Graves et al., 2006), established the dominant paradigm for HTR for nearly a decade. On IAM, vanilla CRNN-CTC baselines achieve 8–12% CER without language model rescoring.

More recent CRNN variants with residual blocks and an auxiliary CTC shortcut (Retsinas et al., 2022) reach 5.14% CER on IAM line-level recognition without external language models. The gap from our result (~7.5% greedy) to this level is primarily attributable to their CTC shortcut and residual CNN design, both orthogonal to our data-centric contributions.

The current state of the art is dominated by Transformer-based approaches: TrOCR (Li et al., 2021) achieves 2.89% CER by combining a pretrained BEiT image encoder with a RoBERTa decoder, pretrained on 684 million synthetic lines. DRetHTR (2025) further improves to 2.26% CER using a decoder-only RetNet architecture. These models require 200–600 GPU hours for fine-tuning, making controlled data-centric studies difficult. We intentionally use CRNN-CTC to maintain experimental tractability.

### 2.2 Label Noise in Benchmark Datasets

Label noise in training data is a well-documented source of model degradation (Frenay & Verleysen, 2014). In NLP benchmarks, CoNLL-03 NER has been estimated to contain 5–7% label noise (Wang et al., 2019); similar issues appear in OntoNotes4 (~8%) and WNUT-17 (~18%). In image classification, CIFAR-N (Wei et al., 2022) provides human-annotated noisy labels to study the practical impact of real-world annotation errors.

Most relevant to our work, REVEAL (Jiang et al., 2025) proposes a unified framework using multiple VLMs to renovate image classification test sets by detecting label noise and imputing missing labels. We apply a similar VLM-based auditing philosophy to the HTR domain, but add a key finding absent from the classification setting: the VLM prompt design — specifically the bias toward flagging vs. accepting annotations — critically determines whether cleaning helps or hurts downstream training.

To our knowledge, no prior work has systematically audited IAM annotation quality using modern VLMs, nor demonstrated the sensitivity of data cleaning effects to VLM prompt aggressiveness.

### 2.3 Synthetic Data Augmentation for HTR

Font-based text rendering (trdg, Belval 2019) provides a low-cost source of labeled training data. Prior work has used trdg to generate up to 2.5 million synthetic handwriting lines for pretraining (Luo et al., 2023), with consistent but diminishing gains. Wigington et al. (2017) demonstrated modest but consistent improvement on IAM with image-level augmentation.

A key limitation of font-based synthesis for IAM is the domain gap: rendered text lacks stroke-level variability. We design our pipeline to specifically target underrepresented characters in the IAM distribution (digits, uppercase, punctuation), using an LLM to generate text templates dense in rare characters — addressing known class imbalance rather than simply scaling volume. Our per-character analysis (Section 7.3) confirms this targeted approach produces larger gains on rare characters than on common ones.

### 2.4 Language Model Rescoring for CTC

CTC beam search with n-gram language model rescoring is the standard post-processing approach for both ASR (Hannun et al., 2014) and HTR. The pyctcdecode + KenLM pipeline scores hypotheses as:

```
score = α × CTC_log_prob + β × LM_score + γ × word_count
```

where α, β, γ are tunable weights. Prior work (Hannun et al., 2014) shows that word-level n-gram LMs provide stronger rescoring than character-level or unigram models, because they capture word-level context (e.g., distinguishing `form` from `from`) that CTC's frame-level modeling misses. Our results confirm a 1.12% absolute CER reduction from 4-gram LM rescoring vs. greedy decoding (7.53%→6.41%), and a 0.09% reduction vs. unigram beam search (7.50%→6.41%).

---

## 3. Dataset and Preprocessing

### 3.1 IAM Handwriting Database

The IAM Handwriting Database (Marti & Bunke, 2002) contains handwritten English text scanned at 300 dpi from 657 writers, originally sourced from the LOB corpus. We use the standard line-level split:

| Split | Samples | Writers |
|---|---|---|
| Train | 6,482 | ~547 |
| Validation | 976 | ~55 |
| Test | 2,915 | ~55 |
| **Total** | **10,373** | **657** |

**Preprocessing:** All images are resized to height H=64 pixels with aspect ratio preserved via padding. Images are normalized per-channel (zero mean, unit variance) and stored in LMDB format for efficient batch loading. The character vocabulary contains 80 classes: 79 IAM-standard characters plus a CTC blank token (index 0):

```
 !"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz
```

### 3.2 VLM Annotation Quality Audit (V1 — Aggressive)

We first audited all 10,373 samples using Doubao Seed 2.0 Pro with a prompt instructing the model to "err on the side of flagging" any possible annotation error. The VLM output a structured verdict (CORRECT / INCORRECT / AMBIGUOUS).

**V1 audit results:**

| Split | Total | Flagged | Flag Rate |
|---|---|---|---|
| Train | 6,482 | 2,144 | 33.1% |
| Validation | 976 | 360 | 36.9% |
| Test | 2,915 | 1,053 | 36.1% |
| **Total** | **10,373** | **3,557** | **34.3%** |

The high flag rate (~34%) reflects a combination of genuine transcription errors and VLM over-flagging: synonym substitutions (`out-dated` → `outmoded`), grammar corrections, and stylistic rewrites. Example flagged entries:

| Ground Truth | VLM Correction | Type |
|---|---|---|
| `0M P for Manchester Exchange .` | `UM P for Manchester Exchange .` | Likely genuine |
| `meeting of Labour 0M Ps tommorow` | `meeting of Labour OM Ps tomorrow` | Mixed |
| `which would appear to "prop up" an out-dated` | `which would appear to "prop up" an outmoded` | VLM over-flag |
| `and he is to be backed by Mr. Will` | `and her is to be backed by her Will` | VLM hallucination |

### 3.3 VLM Annotation Quality Audit (V2 — Conservative)

To investigate the effect of audit aggressiveness, we designed an improved prompt (v2) that explicitly instructs the model to "default to CORRECT" and lists specific false-positive traps (digit 0 vs. letter O, British spelling variants, abbreviations, minor punctuation differences, handwriting style variations). V2 was applied to the training split only.

**V2 audit results (train split):**

| | V1 Prompt | V2 Prompt |
|---|---|---|
| Flagged samples | 2,144 (33.1%) | 814 (12.6%) |
| Resulting train set | 4,338 | 5,668 |

The V2 prompt reduces the flag rate from 33.1% to 12.6%, removing 1,330 fewer samples by applying a stricter standard for what constitutes a genuine annotation error.

### 3.4 Character Confusion Analysis

Substitutions were extracted by character-level alignment (`difflib.SequenceMatcher`) between IAM ground-truth and VLM V1-corrected text across all flagged samples. We identified **936 unique confusion pairs** totaling **5,058 substitution events** across train + val + test.

**Top 20 confusion pairs:**

| Rank | IAM label | VLM correction | Count | Category |
|---|---|---|---|---|
| 1 | `o` | `a` | 155 | lower–lower |
| 2 | `n` | `m` | 135 | lower–lower |
| 3 | `r` | `s` | 133 | lower–lower |
| 4 | `a` | `o` | 116 | lower–lower (symmetric) |
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
- `o` ↔ `a` is the dominant bidirectional pair (271 combined events)
- `n` is a confusion hub: confused with `m` (135), `u` (96), `r` (56), `s` (36)
- `t` is similarly problematic: confused with `h` (75), `d` (68), `l` (54)
- Digits and uppercase letters are rarely confused, suggesting annotators had greater difficulty with cursive lowercase than with other character classes

The `vlm_confusion_multi.csv` file additionally records multi-character alignment patterns (2→1 merges such as `th→H`, 1→2 splits such as `d→cl`), capturing 2,401 unique multi-character confusion events used in extended analyses.

### 3.5 Cleaned Training Sets

| Set | Samples | Construction |
|---|---|---|
| `train` (full) | 6,482 | Original IAM |
| `train_clean` (V1) | 4,338 | Removed 2,144 V1-flagged (−33%) |
| `train_clean_2` (V2) | 5,668 | Removed 814 V2-flagged (−12.6%) |

### 3.6 IAM Character Distribution

The IAM training set has a heavily skewed character distribution. Lowercase letters account for ~68% of all characters; digits, uppercase, and punctuation are significantly underrepresented, with characters `#`, `*`, `+`, `Z`, `Q` appearing fewer than 50 times total. This imbalance motivates our targeted synthetic data pipeline (Section 5).

---

## 4. Model Architecture

### 4.1 CRNN-CTC

Our model follows the standard CRNN pipeline: CNN feature extractor → BiGRU encoder → CTC decoder.

**CNN Feature Extractor (H=64 adapted)**

We adapt the original VGG CRNN backbone (designed for H=32) to H=64 by modifying the final convolutional kernel:

| Layer | Kernel / Pool | Output (H×W×C) | BN |
|---|---|---|---|
| Conv1 + Pool | 3×3, MaxPool 2×2 | 32×W/2×64 | No |
| Conv2 + Pool | 3×3, MaxPool 2×2 | 16×W/4×128 | No |
| Conv3 + Conv4 + Pool | 3×3, AsymPool | 4×W/4×256 | After Conv3 |
| Conv5 + Conv6 + Pool | 3×3, AsymPool | 2×W/4×512 | After Conv5 |
| Conv7 | kernel=(4,1) | 1×W/4×512 | After Conv7 |

The asymmetric pooling in layers 3–4 preserves horizontal resolution while halving vertical resolution, which is standard practice for text line recognition. The final kernel=(4,1) (vs. (2,1) in the original for H=32) is validated to outperform the original at H=64 (7.35% vs. 7.45% val CER at 50 epochs).

**Sequence Encoder:** 2-layer Bidirectional GRU, hidden size = 512. Input size = 512.

**Output projection:** Linear(1024 → 80), log-softmax over 80 classes (79 characters + blank).

**Loss:** CTCLoss (blank=0, zero_infinity=True, reduction='mean')

**Total parameters: ~16.7M**

### 4.2 Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, weight_decay=1e-4 |
| Initial LR | 3×10⁻⁴ |
| LR schedule | ReduceLROnPlateau (patience=10, factor=0.8, min_lr=1e-7) |
| Warmup | 1 epoch linear from ≈0 to initial LR |
| Batch size | 64 |
| Gradient clip | L2 norm ≤ 10.0 |
| Dropout | 0.1 (before BiGRU) |
| Epochs | 150 (main) / 50 (hyperparameter proxy) |
| Input height | H=64, width variable (aspect ratio preserved) |
| Augmentation | Random affine, elastic distortion, brightness jitter (training only) |

---

## 5. Synthetic Data Pipeline

### 5.1 Motivation

The IAM training set contains only 6,482 samples with severe character imbalance: rare characters such as `#`, `*`, `;` appear fewer than 10–60 times. Font-based synthesis can efficiently increase sample count and character coverage without human annotation.

### 5.2 Text Sources

We use a three-layer text pool:

**Layer 1 — IAM training labels (×2, ~13,000 sentences)**
Direct re-use ensures vocabulary and sentence structure closely match the test distribution.

**Layer 2 — LLM-generated rare-character sentences (~12,000)**
We prompt Doubao Seed 2.0 Pro to generate sentences where each target character appears at least 2 times. Target characters: all digits, uppercase letters with < 500 IAM training occurrences, and rare punctuation (`#`, `&`, `*`, `+`, `/`, `;`, `:`).

Example LLM-generated sentences targeting `;`:
```
The proposal covers three areas: transport, housing; and public health funding.
Section 4; paragraph 7 outlines the terms of service; refer there first.
```

**Layer 3 — Template sentences (~3,000)**
Rule-based templates with random slots for digits, names, dates, and punctuation, serving as a lightweight fallback.

**Combined pool: ~28,000 sentences**, shuffled before generation.

### 5.3 Rendering

Font rendering uses trdg (Belval, 2019) with 14 fonts: 9 handwriting-style (ArchitectsDaughter, HomemadeApple, IndieFlower, Kalam-Light, Kalam-Regular, PatrickHand, PermanentMarker, RockSalt, Satisfy) and 5 print fonts (DroidSans, Lato-Regular, Lato-Italic, OpenSans-Regular, Raleway-Regular). Bold fonts were excluded to avoid stroke-width distribution shift.

Each image: H=64px, random skew (0–3°), random blur (0–1px), random distortion (elastic/perspective/none).

| Scale | Synthetic samples | Total with IAM train |
|---|---|---|
| 1× | 6,500 | ~13,000 |
| 2× | 13,000 | ~19,500 |
| 3× | 19,500 | ~26,000 |
| 5× | 32,500 | ~39,000 |
| 7× | 45,500 | ~52,000 |
| 10× | 65,000 | ~71,500 |

A single 65,000-sample LMDB is generated once; training experiments use subsets with a fixed random seed to ensure comparability.

---

## 6. Hyperparameter Search

### 6.1 Protocol

Greedy sequential search: starting from a baseline, change one hyperparameter at a time and keep the change if validation CER improves by > 0.05% relative. All runs use 50 epochs as a proxy budget (vs. 150 for main experiments).

**Baseline:** hidden=256, dropout=0.1, batch_size=64, lr=1e-4, cnn_out=512, weight_decay=1e-4

### 6.2 Results

| Run | Change | Val CER | Δ | Decision |
|---|---|---|---|---|
| 00 | Baseline (hidden=256) | 9.04% | — | Keep |
| 01 | hidden=128 | 9.79% | +0.75% | Discard |
| **02** | **hidden=512** | **7.35%** | **−1.69%** | **Keep** |
| 03 | dropout=0.0 | 7.45% | +0.10% | Discard |
| 04 | dropout=0.2 | 7.85% | +0.50% | Discard |
| 05 | dropout=0.3 | 7.20% | −0.15% | Keep |
| 06 | batch_size=32 | 6.73% | −0.62% | Keep |
| 07 | batch_size=128 | 8.39% | +1.04% | Discard |
| **08** | **lr=3×10⁻⁴** | **6.29%** | **−1.06%** | **Keep** |
| 09 | lr=5×10⁻⁵ | 9.00% | +2.71% | Discard |
| 10 | cnn_out=256 | 7.50% | +1.21% | Discard |
| 11 | weight_decay=1e-3 | 8.05% | +1.76% | Discard |
| 12 | weight_decay=0 | 7.42% | +1.13% | Discard |
| 13 | original VGG kernel | 7.45% | +1.16% | Discard |

**Optimal:** hidden=512, lr=3×10⁻⁴, dropout=0.1, batch_size=64, cnn_out=512, weight_decay=1e-4. **50-epoch proxy CER: 6.29%**

### 6.3 Analysis

**hidden=512** produced the largest single improvement (9.04% → 7.35%), indicating the BiGRU was the primary bottleneck. **lr=3×10⁻⁴** provided the second-largest gain (7.35% → 6.29%), reflecting faster convergence within the 50-epoch proxy budget. **batch_size=32** showed modest improvement (7.35% → 6.73%), consistent with the known benefit of noisier gradients for small datasets. **lr=5×10⁻⁵** (9.00%) was the worst result: too-small learning rate makes 50 epochs insufficient for convergence.

---

## 7. Experiments

### 7.1 Experimental Setup

All main experiments use the optimal configuration from Section 6, trained for **150 epochs** with ReduceLROnPlateau and 1-epoch linear warmup.

**Evaluation metric:** Character Error Rate (CER) = edit_distance(hyp, ref) / max(len(ref), 1) × 100%

Test evaluation uses the best-validation-CER checkpoint only. The test set is held out and evaluated once per experiment.

### 7.2 Full vs. VLM-Cleaned IAM (Exp1, Exp2, Exp5)

| Experiment | Training Set | Samples | Test CER | Δ vs. Exp1 |
|---|---|---|---|---|
| **Exp1** — Full IAM | `train` | 6,482 | **8.43%** | — |
| **Exp2** — V1 cleaned | `train_clean` | 4,338 | **10.33%** | +1.90% ↑ worse |
| **Exp5-base** — V2 cleaned | `train_clean_2` | 5,668 | **8.36%** | −0.07% ≈ same |

**Finding:** V1 cleaning (−33% of training data) significantly degrades test CER from 8.43% to 10.33%. This result is counterintuitive — removing supposedly noisy samples makes the model worse. V2 cleaning (−12.6%) produces a negligible change (8.43% → 8.36%), suggesting that the cleaned samples are broadly neutral in their effect.

Two explanations account for V1's regression: (1) the aggressive V1 prompt removes many valid samples alongside genuine errors, reducing effective training set size from 6,482 to 4,338 and causing underfitting; (2) some IAM annotation ambiguities (e.g., digit 0 vs. letter O) that V1 flags are actually correct per IAM convention, and the model needs to see these cases to learn the correct label assignment.

The V2 result shows that a more precise VLM prompt can maintain comparable performance, suggesting the upper bound on cleaning benefit may be achievable with higher VLM precision — though even V2-cleaned training does not outperform the full dataset baseline.

### 7.3 Synthetic Data Augmentation (Exp3, Exp4, Exp5)

| Experiment | Training Set | Synth Scale | Total Samples | Test CER |
|---|---|---|---|---|
| **Exp1** | Full IAM | 0× | 6,482 | 8.43% |
| **Exp3-1x** | Full IAM + synth | 1× | ~13,000 | 7.93% |
| **Exp3-2x** | Full IAM + synth | 2× | ~19,500 | 7.87% |
| **Exp3-3x** | Full IAM + synth | 3× | ~26,000 | 7.66% |
| **Exp3-5x** | Full IAM + synth | 5× | ~39,000 | 7.63% |
| **Exp3-7x** | Full IAM + synth | 7× | ~52,000 | 7.74% |
| **Exp3-10x** | Full IAM + synth | 10× | ~71,500 | **7.52%** |
| **Exp4-1x** | V1 cleaned + synth | 1× | ~10,838 | 8.94% |
| **Exp4-2x** | V1 cleaned + synth | 2× | ~17,338 | 8.76% |
| **Exp4-3x** | V1 cleaned + synth | 3× | ~23,838 | 8.61% |
| **Exp4-5x** | V1 cleaned + synth | 5× | ~36,190 | 8.86% |
| **Exp4-7x** | V1 cleaned + synth | 7× | ~34,866 | 8.91% |
| **Exp4-10x** | V1 cleaned + synth | 10× | ~47,718 | 9.07% |
| **Exp5-1x** | V2 cleaned + synth | 1× | ~12,168 | 8.08% |
| **Exp5-2x** | V2 cleaned + synth | 2× | ~18,668 | 7.97% |
| **Exp5-3x** | V2 cleaned + synth | 3× | ~25,168 | 8.01% |
| **Exp5-5x** | V2 cleaned + synth | 5× | ~38,168 | 7.92% |
| **Exp5-7x** | V2 cleaned + synth | 7× | ~51,668 | 8.17% |
| **Exp5-10x** | V2 cleaned + synth | 10× | ~71,168 | 8.06% |

**Finding:** Synthetic data provides consistent improvement for the full IAM baseline (Exp3): CER decreases from 8.43% to 7.52% at 10× scale, a 10.8% relative improvement. Exp4 (V1-cleaned + synth) shows that even with synthetic augmentation, aggressive cleaning remains harmful — Exp4-3x achieves only 8.61% vs. Exp3-3x's 7.66%, and remarkably, Exp4 CER *increases* beyond 3× (8.61% → 8.91% at 7× → 9.07% at 10×), suggesting the model overfits to the synthetic domain once the V1-cleaned IAM distribution is too small to anchor it. V2-cleaned + synth (Exp5) narrows the gap to ~0.4% below Exp3 at equal scale but still underperforms the full dataset, and similarly stagnates beyond 5×. This divergence from Exp3's improving trend suggests that the VLM-cleaned training distribution has a ceiling around 7.9–8.3% CER regardless of synthetic volume, likely because the model no longer learns from the 814 ambiguous-but-informative samples that V2 cleaning discarded.

### 7.4 Per-Character Error Analysis

We align model outputs to ground-truth using Levenshtein backtrace to compute per-character CER. Table 2 compares Exp1 (full IAM baseline) vs. Exp3-10x (best greedy model) on the test set.

| Character | Count | Exp1 CER | Exp3-10x CER | Δ |
|---|---|---|---|---|
| `#` | 5 | 100.0% | 60.0% | −40.0% |
| `;` | 60 | 41.7% | 21.7% | −20.0% |
| `U` | 11 | 36.4% | 18.2% | −18.2% |
| `R` | 67 | 35.8% | 20.9% | −14.9% |
| `:` | 29 | 55.2% | 41.4% | −13.8% |
| `!` | 47 | 38.3% | 25.5% | −12.8% |
| `0` | 34 | 38.2% | 26.5% | −11.8% |
| `W` | 122 | 32.0% | 20.5% | −11.5% |
| `L` | 123 | 33.3% | 24.4% | −8.9% |
| `F` | 84 | 28.6% | 20.2% | −8.3% |
| `"` | 576 | 27.1% | 17.9% | −9.2% |
| `k` | 726 | 25.5% | 20.0% | −5.5% |
| `z` | 45 | 35.6% | 40.0% | +4.4% |
| `D` | 92 | 25.0% | 26.1% | +1.1% |

*Table 2: Per-character CER comparison. Characters with count < 5 excluded.*

**Finding:** The largest improvements are concentrated precisely in the rare character categories targeted by our synthetic data pipeline: uppercase letters with low IAM frequency (`U`, `R`, `W`, `L`, `F`), digits (`0`), and punctuation (`;`, `:`, `!`, `#`, `"`). This directly confirms that the LLM-generated rare-character text in Layer 2 of our synthesis pipeline (Section 5.2) is effective in its intended purpose.

`z` shows a small regression (+4.4%), plausibly because handwriting-style font renderings of `z` differ visually from IAM's cursive `z`, introducing distribution shift for this character.

### 7.5 VLM Annotation Confusion vs. Model Prediction Confusion

A key question is whether the character confusion patterns identified by the VLM audit (Section 3.4) reflect genuine visual ambiguity or VLM-specific perception biases. To investigate, we compare the VLM annotation confusion matrix against the model's substitution confusion matrix — both derived from character-level Levenshtein alignment.

**VLM confusion:** pairs where the VLM flagged IAM's label as wrong (IAM label → VLM correction), from 5,058 substitution events across all splits.

**Model confusion:** pairs where Exp3-10x predicted the wrong character on the test set, from character-level alignment of predictions vs. ground truth.

Table 3 shows the top-15 shared substitution pairs (appearing in both matrices with count ≥ 5):

| Pair | VLM count | Model count | In both |
|---|---|---|---|
| `o` ↔ `a` | 155 + 116 = 271 | 174 + 280 = 454 | ✓ #1 |
| `n` ↔ `m` | 135 + ? = 135+ | 105 + 89 = 194 | ✓ #2 |
| `r` ↔ `s` | 133 + 59 = 192 | 121 + 92 = 213 | ✓ #3 |
| `a` ↔ `e` | 109 + 46 = 155 | 96 + 56 = 152 | ✓ #4 |
| `n` ↔ `u` | 96 + ? | 58 + 102 = 160 | ✓ |
| `o` ↔ `e` | 76 | 67 + 48 = 115 | ✓ |
| `t` ↔ `l` | 54 | 116 + 49 = 165 | ✓ |
| `h` ↔ `l` | 60 | 52 | ✓ |
| `r` ↔ `n` | 56 + 41 = 97 | 76 + 66 = 142 | ✓ |
| `u` ↔ `e` | 52 | 45 | ✓ |
| `t` ↔ `d` | 68 | 38 | ✓ |
| `s` ↔ `r` | 59 | 92 | ✓ |
| `e` ↔ `a` | 46 | 56 | ✓ |
| `d` ↔ `c` | 39 | 15 | ✓ |

*Table 3: Top shared confusion pairs between VLM annotation audit and model predictions.*

Across all 160 pairs appearing in both matrices with count ≥ 5, the Spearman rank correlation between VLM confusion counts and model confusion counts is **ρ = 0.559 (p < 0.0001)**.

**Interpretation:** The significant positive correlation indicates that both the VLM and the trained model struggle with the same character pairs — suggesting these pairs represent genuine visual ambiguity in handwritten English text rather than idiosyncratic biases of either system. The pairs `o↔a`, `n↔m`, `r↔s` are systematically hard to disambiguate for both human-trained annotators (as audited by the VLM) and machine learning models alike.

This convergence has a practical implication: **character confusion patterns from VLM annotation auditing can serve as a prior for targeted data collection**. Characters that appear frequently in both matrices (particularly `o`, `a`, `n`, `m`, `r`, `s`) are those where additional training examples with clear, unambiguous handwriting would most benefit model performance — a more principled approach to data augmentation than uniform scaling.

---

## 8. Language Model Rescoring

### 8.1 KenLM 4-gram Language Model

We train a word 4-gram KenLM language model using `lmplz` with modified Kneser-Ney smoothing on a corpus of 183,468 sentences comprising:
- IAM train + validation text labels (~7,400 sentences)
- 176,010 lines from 20 Project Gutenberg public-domain English novels (Pride and Prejudice, Moby Dick, Frankenstein, Sherlock Holmes, and 16 others)

The resulting model file is 150MB (ARPA format). Beam search decoding uses pyctcdecode with the 4-gram model and a word vocabulary of 73,540 words.

### 8.2 Results

| Decoding | Model | Greedy CER | Beam CER | Δ |
|---|---|---|---|---|
| CTC greedy | Exp3-10x | 7.53% | — | — |
| Beam (lowercased unigram vocab) | Exp3-10x | — | 7.50% | −0.03% |
| Beam (4-gram KenLM, lowercased vocab) | Exp3-10x | — | 6.64% | −0.89% |
| **Beam (4-gram KenLM, mixed-case vocab)** | **Exp3-10x** | — | **6.41%** | **−1.12%** |
| Beam (4-gram KenLM, mixed-case) | Exp3-5x | 7.65% | 6.56% | −1.09% |
| Beam (4-gram KenLM, mixed-case) | Exp3-3x | 7.67% | 6.61% | −1.07% |
| Beam (4-gram KenLM, mixed-case) | Exp3-1x | 7.92% | 6.58% | −1.34% |
| Beam (4-gram KenLM, mixed-case) | Exp5-5x | 7.92% | 6.90% | −1.02% |
| Beam (4-gram KenLM, mixed-case) | Exp5-7x | 8.17% | 6.84% | −1.33% |
| Beam (4-gram KenLM, mixed-case) | Exp5-10x | 8.06% | 6.87% | −1.19% |
| Beam (4-gram KenLM, mixed-case) | Exp4-3x | 8.61% | 7.37% | −1.24% |
| Beam (4-gram KenLM, mixed-case) | Exp4-10x | 9.07% | 7.64% | −1.44% |

*Table 3: Beam search decoding results (beam=50, α=0.5, β=1.0)*

**Finding:** The initial unigram vocabulary was built by lowercasing all IAM words, which caused proper nouns (`London`, `Mr`, `Manchester`, `Evans`) to be treated as unknown tokens and penalized by `unk_score_offset=-10.0` during beam search. Rebuilding the vocabulary with original case improved CER from 6.64% to 6.41%.

A striking cross-row observation: **Exp3-1x + beam (6.58%)** is within 0.17% of **Exp3-10x + beam (6.41%)**, even though Exp3-1x requires only 1/10th the synthetic data. This suggests that for models already benefiting from beam search, additional synthetic data beyond 1× provides rapidly diminishing returns — LM rescoring is more cost-efficient than scaling synthetic volume once a reasonable greedy baseline is established. Conversely, Exp4 (V1-cleaned) models receive larger absolute beam improvements (−1.24% to −1.44%) precisely because their weaker greedy baselines have more recoverable errors via LM context; yet their post-beam CER (7.37–7.64%) remains well above Exp3.

The best result in our study is **exp3-10x + 4-gram KenLM (mixed-case vocab): 6.41% test CER**, combining synthetic data augmentation and language model rescoring.

---

## 9. Discussion

### 9.1 The Critical Role of VLM Prompt Design in Annotation Cleaning

Our most important finding is that VLM-based annotation cleaning is highly sensitive to prompt aggressiveness. V1 ("err on the side of flagging") removes 33% of training data and degrades performance by 1.90% absolute. V2 ("default to CORRECT") removes only 12.6% and produces negligible change. Neither prompt enables cleaning to improve over the full dataset.

The cross-matrix analysis (Section 7.5) sheds light on why: the top confusion pairs flagged by the VLM (`o↔a`, `n↔m`, `r↔s`) are the same pairs where the model makes the most prediction errors. This means the VLM and the model share the same perceptual difficulty — both genuinely cannot reliably distinguish these cursive strokes. When the V1 prompt removes IAM samples containing these ambiguous characters, it removes difficult-but-correctly-labeled examples that the model needs to learn from. The "noise" the VLM is trying to remove is largely indistinguishable from hard-but-valid training signal.

This reframes the VLM cleaning failure: it is not simply that the VLM is inaccurate, but that the target ambiguities (`o` vs. `a` in cursive handwriting) are genuinely hard — for human annotators, VLMs, and trained models alike. The Spearman ρ=0.559 correlation between VLM and model confusion patterns quantifies this shared difficulty.

**Practical implication:** VLM-based annotation cleaning for HTR is most credible for clear, high-confidence errors (e.g., a word entirely unrelated to the handwritten content). For the dominant confusion pairs (`o↔a`, `n/m/u`), cleaning requires human adjudication or per-character confidence gating from the model itself, not a binary VLM verdict.

### 9.2 Synthetic Data: Diminishing Returns and Targeted vs. Generic Augmentation

The Exp3 CER curve decreases from 8.43% (0×) to 7.52% (10×) with a roughly monotonic but diminishing trend. The 1× → 3× range provides the most consistent gains (~0.7% total); the 5× → 10× range adds ~0.1% more, consistent with diminishing returns from domain gap effects.

The per-character analysis (Table 2) reveals that synthetic data's primary contribution is repairing rare characters, not improving common ones. For `e`, `a`, `n`, `o` (the dominant characters in IAM), CER changes are small (<2%). For `;`, `R`, `U`, `0`, `!` (low-frequency in IAM), improvements are 10–20%. This asymmetry validates our design choice of using LLM-generated targeted text to address class imbalance directly.

### 9.3 Language Model Rescoring vs. Synthetic Data

Comparing the two post-baseline improvements:

| Method | CER gain (vs. Exp1 baseline) | Implementation cost |
|---|---|---|
| 10× synthetic augmentation | 0.91% (8.43%→7.52%) | High (data generation, extended training) |
| 4-gram KenLM beam search (mixed-case) | 1.12% (7.53%→6.41%) | Low (one-time LM training, no retraining) |

KenLM rescoring slightly outperforms 10× synthetic augmentation while requiring no model retraining — making it the higher-value optimization per unit of effort. In practice, the two are complementary: combining both (Exp3-10x + beam) achieves **6.41%**, compared to 7.52% for augmentation alone.

### 9.4 Limitations

**VLM cleaning does not help.** Neither V1 nor V2 cleaning improves over the full dataset baseline. This does not disprove the hypothesis that IAM contains annotation errors; it shows that our VLM-based approach cannot reliably isolate genuine errors from ambiguous-but-correct annotations at the required precision.

**CRNN-CTC is not state of the art.** Our best result (6.41%) is approximately 3.5% absolute above TrOCR-Large (2.89%). Whether the same data-centric findings hold for Transformer-based HTR remains untested.

**Single dataset.** All experiments are on IAM. Generalizability to RIMES, CVL, or Bentham is not demonstrated.

**Synthetic domain gap.** Font-rendered text lacks stroke-level variability. GAN-based or diffusion-based synthesis would likely yield larger gains at equal volume, at substantially higher computational cost.

---

## 10. Conclusion

We present a data-centric study of HTR on IAM examining the effects of hyperparameter optimization, VLM-based annotation cleaning, synthetic data augmentation, and language model rescoring.

**Key results:** A 14-run hyperparameter search reduces validation CER from 9.04% to 6.29% at 50 epochs, identifying BiGRU hidden size and learning rate as the two most impactful factors. Synthetic data augmentation at 10× scale improves test CER from 8.43% to 7.52%, with per-character analysis confirming targeted improvements in rare uppercase letters and punctuation. VLM annotation cleaning is highly sensitive to prompt design: aggressive cleaning (V1, −33%) degrades CER by 1.90%, while conservative cleaning (V2, −12.6%) has negligible effect. 4-gram KenLM beam search with mixed-case vocabulary reduces CER by 1.12% with no model retraining, slightly outperforming 10× synthetic augmentation at a fraction of the cost. Our best result — 10× synth + 4-gram beam search (mixed-case vocab) — achieves **6.41% test CER**.

Two findings stand out for future HTR research. First, VLM annotation cleaning is prompt-sensitive: the same VLM with different prompts either hurts or is neutral, and neither cleans the benchmark below baseline CER. The cross-matrix correlation (ρ=0.559) between VLM and model confusion patterns reveals why: both systems share the same perceptual difficulty on cursive strokes like `o↔a` and `n↔m`, making it impossible for the VLM to cleanly separate annotation errors from hard-but-valid training examples without per-character confidence gating. Second, the convergence of VLM annotation confusion and model prediction confusion onto the same character pairs suggests a principled strategy for future data collection: targeted acquisition of unambiguous handwriting samples for the high-confusion pairs, rather than uniform dataset scaling.

**Future work:** (1) Manual validation of VLM-flagged samples to quantify true annotation error rate in IAM; (2) Extension of VLM auditing to additional HTR benchmarks (RIMES, CVL) to test generalizability; (3) Integration of model prediction uncertainty to identify samples where VLM correction is most credible; (4) Application of diffusion-based handwriting synthesis for higher-quality rare-character augmentation.

---

## References

Belval, E. (2019). TextRecognitionDataGenerator. GitHub. https://github.com/Belval/TextRecognitionDataGenerator

Frenay, B., & Verleysen, M. (2014). Classification in the presence of label noise: A survey. *IEEE Transactions on Neural Networks and Learning Systems*, 25(5), 845–869.

Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. *ICML 2006*.

Hannun, A., Case, C., Casper, J., Catanzaro, B., Diamos, G., Enberg, T., Prenger, R., Salimans, T., Sanjeev, S., Coates, A., & Ng, A. (2014). Deep Speech: Scaling up end-to-end speech recognition. *arXiv:1412.5567*.

Jiang, X., et al. (2025). When VLMs Meet Image Classification: Test Sets Renovation via VLM-Guided Generative Data Augmentation. *arXiv:2505.16149*.

Kang, L., Rusiñol, M., Fornés, A., Riba, P., & Villegas, M. (2020). Unsupervised adaptation for synthetic-to-real handwritten word recognition. *WACV 2020*.

Li, M., Lv, T., Cui, L., Lu, Y., Florencio, D., Zhang, C., Li, Z., & Wei, F. (2021). TrOCR: Transformer-based optical character recognition with pre-trained models. *arXiv:2109.10282*.

Luo, C., Jin, L., & Sun, Z. (2023). CNN-BiLSTM model for English handwriting recognition: Comprehensive evaluation on the IAM dataset. *arXiv:2307.00664*.

Marti, U.-V., & Bunke, H. (2002). The IAM-database: An English sentence database for offline handwriting recognition. *International Journal on Document Analysis and Recognition*, 5(1), 39–46.

Retsinas, G., Sfikas, P., Gatos, B., & Nikou, C. (2022). Best practices for a handwritten text recognition system. *arXiv:2404.11339*.

Shi, B., Bai, X., & Yao, C. (2016). An end-to-end trainable neural network for image-based sequence recognition. *IEEE TPAMI*, 39(11), 2298–2304.

Wei, J., Zhu, Z., Cheng, H., Liu, T., Niu, G., & Liu, Y. (2022). Learning with noisy labels revisited: A study using real-world human annotations. *ICLR 2022*.

Wigington, C., Stewart, S., Price, B., & Cohen, S. (2017). Data augmentation for recognition of handwritten words and lines using a CNN-LSTM network. *ICDAR 2017*.

---

## Appendix A: Hyperparameter Search Configuration Details

| Run | hidden | dropout | batch | lr | cnn_out | wd |
|---|---|---|---|---|---|---|
| 00 | 256 | 0.1 | 64 | 1e-4 | 512 | 1e-4 |
| 01 | 128 | 0.1 | 64 | 1e-4 | 512 | 1e-4 |
| 02 | 512 | 0.1 | 64 | 1e-4 | 512 | 1e-4 |
| 03 | 512 | 0.0 | 64 | 1e-4 | 512 | 1e-4 |
| 04 | 512 | 0.2 | 64 | 1e-4 | 512 | 1e-4 |
| 05 | 512 | 0.3 | 64 | 1e-4 | 512 | 1e-4 |
| 06 | 512 | 0.1 | 32 | 1e-4 | 512 | 1e-4 |
| 07 | 512 | 0.1 | 128 | 1e-4 | 512 | 1e-4 |
| 08 | 512 | 0.1 | 64 | 3e-4 | 512 | 1e-4 |
| 09 | 512 | 0.1 | 64 | 5e-5 | 512 | 1e-4 |
| 10 | 512 | 0.1 | 64 | 1e-4 | 256 | 1e-4 |
| 11 | 512 | 0.1 | 64 | 1e-4 | 512 | 1e-3 |
| 12 | 512 | 0.1 | 64 | 1e-4 | 512 | 0.0 |
| 13 | 512 | 0.1 | 64 | 1e-4 | 512 | 1e-4 (orig VGG) |

## Appendix B: VLM Audit Prompt Comparison

**V1 Prompt (aggressive — flag rate 33%):**
```
The image shows a single line of handwritten English text.
The claimed annotation is: "{annotation}"
Compare word by word. Reply CORRECT / INCORRECT / AMBIGUOUS.
Rules:
- INCORRECT: use whenever you can read a word and it differs. Err on the side of flagging.
- AMBIGUOUS: only when ink is too faded to read at all.
```

**V2 Prompt (conservative — flag rate 12.6%):**
```
The image shows a single line of handwritten English text from the IAM
Handwriting Database, annotated by trained human experts (expected ~95%+ accuracy).
Annotation: "{annotation}"
Default to CORRECT. Only flag INCORRECT when ALL of the following:
  - You can clearly read the word(s) in question
  - The difference is unambiguous, not a style variation
  - You can specify exactly which characters differ
Do NOT flag: digit 0 vs letter O (IAM uses 0 for zero; trust annotation),
minor punctuation, British spelling, abbreviations, proper nouns,
handwriting style variations.
```

The V2 prompt's specific enumeration of false-positive traps is the primary driver of the 20.5% reduction in flag rate, with the "default to CORRECT" framing reducing hallucinated corrections.
