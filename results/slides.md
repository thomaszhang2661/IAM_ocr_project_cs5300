---
marp: true
theme: default
paginate: true
header: "CS5300 Computer Vision · Final Project · April 2026"
footer: "Jian (Thomas) Zhang · Northeastern University"
style: |
  section {
    font-size: 22px;
  }
  section.title {
    text-align: center;
    justify-content: center;
  }
  h1 { color: #c00000; }
  h2 { color: #c00000; }
  table { font-size: 18px; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }
  .highlight { background: #fff3cd; padding: 4px 8px; border-radius: 4px; }
  blockquote { border-left: 4px solid #c00000; background: #f8f8f8; font-size: 18px; }
---

<!-- _class: title -->

# Handwritten Text Recognition on IAM
## VLM Annotation Auditing · Synthetic Augmentation · LM Rescoring

**Jian (Thomas) Zhang**
MS Computer Science (Align) · Northeastern University
CS5300 Computer Vision — Final Project | April 2026

---

<!-- Slide 2: Problem & Approach — ~2 min -->

## Motivation & Research Questions

**Task:** Transcribe handwritten line images → digital text (HTR)

![bg right:38% 95%](results/test_samples.png)

**Dataset:** IAM Handwriting Database
- 13,353 line images, 657 writers
- Most widely used English HTR benchmark

**Architecture:** CRNN-CTC (BiGRU, 16.7M params)
→ Controlled setting to study **data-centric** effects

**Four research questions:**
| | Question |
|---|---|
| RQ1 | Does hyperparameter search substantially improve CRNN-CTC? |
| RQ2 | Does VLM annotation cleaning help — and how sensitive is prompt design? |
| RQ3 | Does synthetic augmentation improve rare-character recognition? |
| RQ4 | Does n-gram LM rescoring provide meaningful CER gains? |

<!-- 
Speaker notes:
- Start by showing what HTR looks like — the image on the right shows actual IAM samples
- Emphasize this is NOT about beating Transformers — it's about understanding what data choices matter
- CRNN-CTC is our controlled lab environment
-->

---

<!-- Slide 3: Architecture — ~1.5 min -->

## System Architecture: CRNN-CTC

```
Input image (H=64)
    │
    ▼
┌─────────────────────────────┐
│  VGG CNN (7 layers, 512ch)  │  ← spatial feature extraction
└─────────────────────────────┘
    │  mean-pool over height
    ▼
┌─────────────────────────────┐
│  BiGRU × 2  (hidden=512)    │  ← sequence modeling, 16.7M params
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Linear → CTC decoder       │  ← 81 classes (80 chars + blank)
└─────────────────────────────┘
    │
    ▼
  Transcript
```

**Training:** CTC loss · Adam · lr=3×10⁻⁴ · weight decay · dropout 0.1
**Alphabet:** 80 characters — letters, digits, punctuation

<!-- 
Speaker notes:
- CTC = Connectionist Temporal Classification: handles variable-length alignment without segmentation
- BiGRU reads the feature sequence left→right and right→left simultaneously
- blank token allows CTC to output "no character here" — enables alignment
-->

---

<!-- Slide 4: Contribution 1 — ~2 min -->

## C1: Hyperparameter Search

**14 parallel runs on 7 GPUs** — grid search over 4 dimensions

![bg right:40% 95%](results/hparam_search_chart.png)

**Search space:**
- RNN type: LSTM vs BiGRU
- Hidden size: 256 / 512
- Learning rate: 1e-3 / 3e-4 / 1e-4
- Input height: 32 / 64

**Result:** BiGRU + hidden=512 + lr=3×10⁻⁴ + H=64

| Config | Val CER |
|---|---|
| Baseline (LSTM, h=256) | 9.04% |
| **Best (BiGRU, h=512)** | **6.29%** |
| Relative improvement | **30%** |

> **Key finding:** BiGRU outperforms LSTM; hidden size matters more than learning rate at this scale.

<!-- 
Speaker notes:
- All 14 runs started simultaneously on different GPUs
- Total search cost: ~3 GPU-days, but only 6 hours wall-clock
- The 30% improvement comes almost entirely from BiGRU + hidden=512 — these two dominate
-->

---

<!-- Slide 5: Contribution 2 — ~3 min -->

## C2: VLM Annotation Auditing

**Doubao Seed 2.0 Pro** audits all 10,373 IAM training samples

<div class="columns">
<div>

**Two prompts tested:**

| | V1 (Aggressive) | V2 (Conservative) |
|---|---|---|
| Bias | "err on flagging" | "default CORRECT" |
| Removed | 33% of data | 12.6% |
| Test CER | 10.33% ↑ | 8.36% ≈ |

**V1 hurts badly. V2 is neutral.**

</div>
<div>

![90%](results/vlm_confusion_matrix_1to1.png)

*VLM annotation confusion matrix*

</div>
</div>

**Novel finding — Cross-Matrix Correlation:**

Model prediction confusions vs. VLM annotation confusions → **Spearman ρ = 0.559** (p < 0.0001, n=160 pairs)

> Both the VLM and the model struggle with **the same character pairs** (`o↔a`, `n↔m`, `r↔s`) — because cursive handwriting is genuinely ambiguous, not because of annotation error.

<!-- 
Speaker notes:
- V1 removes 33% of data — that's enormous. Model starves.
- V2 is more surgical but still neutral — neither prompt can beat the full dataset
- The cross-matrix correlation is the most interesting scientific finding:
  The VLM "errors" are NOT annotation errors — they're genuinely hard characters that even a trained model confuses
  This explains WHY cleaning can't help: you'd need per-character confidence gating, not a binary VLM verdict
-->

---

<!-- Slide 6: Contribution 3 — ~2.5 min -->

## C3: Synthetic Data Augmentation

**Targeting rare characters** — font rendering with TRDG

![bg right:35% 92%](results/synth_samples.png)

**Rare-character oversampling:** generate synthetic lines for characters in the bottom 10% of frequency (`#`, `!`, `;`, `:`, uppercase)

**Scale experiment (Full IAM + N× synth):**

| Scale | Total samples | Test CER |
|---|---|---|
| 0× (Exp1 baseline) | 6,482 | 8.43% |
| 1× | ~13,000 | 7.93% |
| 5× | ~39,000 | 7.63% |
| **10×** | **~71,500** | **7.52%** |

**Per-character analysis** confirms targeted improvement:

| Char | Exp1 CER | Exp3-10x CER | Gain |
|---|---|---|---|
| `#` | 100% | 60% | −40% |
| `;` | 41.7% | 21.7% | −20% |
| `U` | 36.4% | 18.2% | −18.2% |

<!-- 
Speaker notes:
- Synthetic data is font-rendered — much cheaper than real handwriting collection
- The gain is monotonic up to 10×, with diminishing returns after 5×
- Per-character analysis is done via Levenshtein backtrace — we align predictions to ground truth and count substitutions per character class
- Key insight: gains are largest exactly for the rare characters we targeted — the pipeline works as designed
-->

---

<!-- Slide 7: Contribution 4 — ~2 min -->

## C4: Beam Search + KenLM Language Model

**Training corpus:** 183,468 sentences (IAM labels + 20 Gutenberg novels)
**Model:** Word 4-gram KenLM (150MB) · Vocabulary: 73,540 words

**Beam search pipeline:** pyctcdecode · beam=50 · α=0.5 · β=1.0

| Decoding | CER | Δ |
|---|---|---|
| CTC greedy | 7.53% | — |
| Beam + unigram vocab | 7.50% | −0.03% |
| Beam + 4-gram KenLM (lowercased vocab) | 6.64% | −0.89% |
| **Beam + 4-gram KenLM (mixed-case vocab)** | **6.41%** | **−1.12%** |

**Engineering bug discovered:** Lowercasing all vocabulary words caused proper nouns (`London`, `Evans`, `Manchester`) to be penalized as unknown tokens. Preserving case: −0.23% additional CER.

**Concrete correction example:**
```
REF:  "He writes with Tolchard Evans, composer of Lady of Spain"
GRD:  "He wribes with Tolchard Evons, composer of Lady of Spain"
BEAM: "He writes with Tolchard Evans, composer of Lady of Spain" ✓
```

<!-- 
Speaker notes:
- Unigram vocabulary contributes almost nothing — it's the 4-gram context that matters
- The case-sensitivity bug is a good lesson: always inspect your vocabulary
- The correction example shows the LM doing exactly what we want: fixing "wribes→writes" using word context, and "Evons→Evans" using the vocabulary
- 1.12% absolute improvement with zero model retraining — very cost-effective
-->

---

<!-- Slide 8: Summary & Conclusion — ~2 min -->

## Results & Conclusion

**Cumulative improvement over baseline:**

| Method | Test CER | Improvement |
|---|---|---|
| Exp1: Full IAM, greedy | 8.43% | — |
| + Hyperparameter optimization | 8.43%→val 6.29% | 30% relative (val) |
| + 10× Synthetic augmentation | **7.52%** | −0.91% |
| + 4-gram KenLM beam search | **6.41%** | −1.12% additional |
| **State of art (TrOCR-Large)** | *2.89%* | *reference* |

**Four takeaways:**

1. **Hyperparameter search** yields large gains cheaply — always do it first
2. **VLM cleaning is prompt-sensitive** — same VLM, different prompt → opposite outcomes; cross-matrix ρ=0.559 explains why neither prompt can succeed
3. **Synthetic augmentation** works for rare characters; diminishing returns after 5×
4. **LM rescoring** (1.12% gain) slightly outperforms 10× augmentation (0.91%) at a fraction of the cost — and vocabulary case matters

**Future work:** Auxiliary CTC shortcut (Retsinas 2022), diffusion-based handwriting synthesis, per-character confidence-gated VLM cleaning

<!-- 
Speaker notes:
- Our best result 6.41% vs TrOCR 2.89% — 3.5% gap, expected for CRNN vs. Transformer
- The main contribution is scientific insight, not SOTA numbers
- The cross-matrix finding is novel: no prior work has correlated VLM annotation confusion with model prediction confusion on HTR
- Close by emphasizing the cost-efficiency angle: beam search gives ~same gain as 10× augmentation but requires zero GPU retraining
-->
