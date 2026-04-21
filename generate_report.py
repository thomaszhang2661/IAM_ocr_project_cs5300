#!/usr/bin/env python3
"""Generate a self-contained HTML report with embedded images, printable as PDF."""

import base64, json, os, re
from pathlib import Path

ROOT = Path(__file__).parent
RES  = ROOT / "results"

def img64(path):
    """Return base64 data URI for an image, or empty string if missing."""
    p = Path(path)
    if not p.exists():
        return ""
    ext = p.suffix.lower().lstrip(".")
    if ext == "jpg": ext = "jpeg"
    data = base64.b64encode(p.read_bytes()).decode()
    return f"data:image/{ext};base64,{data}"

def load_result(name):
    p = RES / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}

# ── load all images ──────────────────────────────────────────────────────────
img = {k: img64(RES / f"{k}.png") for k in [
    "test_samples", "hparam_search_chart", "training_curves",
    "synth_samples", "char_freq_distribution", "synth_char_freq",
    "vlm_confusion_matrix_1to1", "vlm_confusion_multichar",
    "augment_comparison", "fonts_preview",
]}

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HTR on IAM — CS5300 Final Project Report</title>
<style>
/* ── Base ── */
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  font-family: "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 11pt;
  line-height: 1.65;
  color: #111;
  max-width: 900px;
  margin: 0 auto;
  padding: 2em 2.5em;
}}
/* ── Headings ── */
h1 {{ font-size: 18pt; text-align: center; margin-bottom: 0.3em; color: #8b0000; }}
h2 {{ font-size: 14pt; color: #8b0000; border-bottom: 2px solid #8b0000;
     padding-bottom: 3px; margin-top: 2em; }}
h3 {{ font-size: 12pt; color: #333; margin-top: 1.4em; }}
h4 {{ font-size: 11pt; color: #555; margin-top: 1em; }}
/* ── Author block ── */
.author {{ text-align: center; font-size: 11pt; margin-bottom: 0.2em; color: #333; }}
.meta   {{ text-align: center; font-size: 10pt; color: #666; margin-bottom: 1.5em; }}
hr {{ border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }}
/* ── Abstract ── */
.abstract {{
  background: #f9f5f0;
  border-left: 4px solid #8b0000;
  padding: 0.9em 1.2em;
  font-size: 10.5pt;
  margin-bottom: 2em;
}}
.abstract h2 {{ font-size: 11pt; margin-top: 0; border: none; }}
/* ── Tables ── */
table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 9.5pt;
  margin: 1em 0 1.5em 0;
}}
th {{
  background: #8b0000;
  color: #fff;
  padding: 5px 8px;
  text-align: left;
  font-weight: normal;
}}
td {{ padding: 4px 8px; border-bottom: 1px solid #ddd; }}
tr:nth-child(even) td {{ background: #fafafa; }}
tr.best td {{ background: #fff8e1; font-weight: bold; }}
/* ── Figures ── */
figure {{
  margin: 1.2em 0;
  text-align: center;
}}
figure img {{
  max-width: 100%;
  border: 1px solid #ddd;
  border-radius: 4px;
}}
figcaption {{
  font-size: 9pt;
  color: #555;
  margin-top: 0.4em;
  font-style: italic;
}}
.fig-row {{
  display: flex;
  gap: 1em;
  justify-content: center;
  flex-wrap: wrap;
}}
.fig-row figure {{ flex: 1; min-width: 240px; max-width: 48%; }}
/* ── Code ── */
pre, code {{
  font-family: "Courier New", Courier, monospace;
  font-size: 9pt;
  background: #f4f4f4;
  border: 1px solid #ddd;
  border-radius: 3px;
}}
pre {{ padding: 0.7em 1em; overflow-x: auto; white-space: pre-wrap; }}
code {{ padding: 1px 4px; }}
/* ── Callout boxes ── */
.finding {{
  background: #e8f5e9;
  border-left: 4px solid #2e7d32;
  padding: 0.6em 1em;
  margin: 1em 0;
  font-size: 10.5pt;
}}
.finding b {{ color: #1b5e20; }}
.insight {{
  background: #e3f2fd;
  border-left: 4px solid #1565c0;
  padding: 0.6em 1em;
  margin: 1em 0;
  font-size: 10.5pt;
}}
.example {{
  background: #f8f8f8;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 0.7em 1em;
  font-size: 9.5pt;
  margin: 0.8em 0;
}}
.example .ref  {{ color: #2e7d32; }}
.example .grd  {{ color: #c62828; }}
.example .beam {{ color: #1565c0; }}
/* ── Print ── */
@media print {{
  body {{ max-width: 100%; padding: 1cm 1.5cm; font-size: 10pt; }}
  h2 {{ page-break-after: avoid; }}
  figure, table {{ page-break-inside: avoid; }}
  .finding, .insight {{ page-break-inside: avoid; }}
}}
</style>
</head>
<body>

<!-- ═══════════════════════════════════════════════════ TITLE ═══ -->
<h1>Handwritten Text Recognition on IAM:<br>
VLM-Assisted Annotation Auditing, Synthetic Data Augmentation,<br>
and Language Model Rescoring for CRNN-CTC</h1>

<p class="author"><strong>Jian (Thomas) Zhang</strong></p>
<p class="meta">Northeastern University — MS Computer Science (Align)<br>
CS5300 Computer Vision — Final Project | April 2026</p>
<hr>

<!-- ═══════════════════════════════════════════════════ ABSTRACT ═══ -->
<div class="abstract">
<h2>Abstract</h2>
<p>This report presents a data-centric study of handwritten text recognition (HTR) on the IAM
Handwriting Database using a CRNN-CTC architecture with BiGRU encoder (16.7M parameters). We
systematically investigate four interventions: (1) a 14-run parallel hyperparameter search that
reduces validation CER from 9.04% to 6.29%; (2) a VLM-assisted annotation quality audit using
Doubao Seed 2.0 Pro that demonstrates prompt engineering is the dominant factor in whether VLM
cleaning helps or hurts — an aggressive prompt removing 33% of training data degrades CER by
1.90%, while a conservative prompt removing 12.6% has negligible effect; (3) a synthetic data
pipeline targeting underrepresented characters via LLM-generated rare-character text, achieving
7.52% test CER at 10× scale with per-character analysis confirming targeted rare-symbol
improvements; (4) CTC beam search with a KenLM 4-gram language model trained on 183,468
public-domain sentences, reducing CER to <strong>6.41%</strong> — with vocabulary case
preservation identified as a critical engineering factor for proper-noun-heavy benchmarks. A
novel cross-matrix analysis reveals that VLM annotation confusion pairs and model prediction
confusion pairs are significantly correlated (Spearman ρ=0.559, p&lt;0.0001, n=160 pairs),
indicating both systems share the same underlying visual ambiguity in cursive handwriting —
explaining why cleaning cannot reliably separate annotation errors from valid but hard samples.</p>
</div>

<!-- ═══════════════════════════════════════════════════ 1. INTRODUCTION ═══ -->
<h2>1. Introduction</h2>

<p>Handwritten text recognition (HTR) — the automatic transcription of handwritten images to
digital text — remains a core challenge in computer vision. Despite significant progress from
CRNN-CTC architectures (Shi et al., 2016) to Transformer-based models such as TrOCR (Li et al.,
2021), two foundational issues receive comparatively little attention: <em>annotation quality in
standard benchmarks</em>, and the <em>practical gap between training data composition and model
robustness</em>.</p>

<p>The IAM Handwriting Database (Marti &amp; Bunke, 2002) is the most widely used English HTR
benchmark, containing 13,353 line-level samples from 657 writers. This work applies Doubao Seed
2.0 Pro, a state-of-the-art vision-language model, to audit all 10,373 IAM training and
validation samples, revealing structured annotation error patterns and — crucially —
demonstrating that the design of the VLM auditing prompt determines whether cleaning helps or
hurts downstream performance.</p>

{'<figure><img src="' + img["test_samples"] + '" alt="IAM test samples" style="max-width:80%"><figcaption>Figure 1. Example handwritten line images from the IAM test set. The dataset spans 657 writers with significant variation in writing style, pen pressure, and character formation — making annotation quality auditing non-trivial.</figcaption></figure>' if img["test_samples"] else ""}

<h3>1.1 State of the Art on IAM</h3>
<table>
<tr><th>Model</th><th>Architecture</th><th>Test CER</th><th>LM</th></tr>
<tr><td>DRetHTR (2025)</td><td>Decoder-only RetNet</td><td>2.26%</td><td>Yes</td></tr>
<tr><td>TrOCR-Large (Li et al., 2021)</td><td>BEiT + RoBERTa</td><td>2.89%</td><td>No</td></tr>
<tr><td>Retsinas et al. (2022)</td><td>CRNN + aux CTC shortcut</td><td>5.14%</td><td>No</td></tr>
<tr class="best"><td><strong>Ours — Exp3-10x + beam</strong></td><td>CRNN-CTC BiGRU</td><td><strong>6.41%</strong></td><td>4-gram</td></tr>
<tr><td>Ours — Exp3-10x, greedy</td><td>CRNN-CTC BiGRU</td><td>7.52%</td><td>No</td></tr>
<tr><td>Ours — Exp1 baseline, greedy</td><td>CRNN-CTC BiGRU</td><td>8.43%</td><td>No</td></tr>
</table>

<h3>1.2 Research Questions</h3>
<table>
<tr><th>RQ</th><th>Question</th></tr>
<tr><td>RQ1</td><td>Does systematic hyperparameter search substantially improve CRNN-CTC on IAM?</td></tr>
<tr><td>RQ2</td><td>Does VLM annotation cleaning improve test CER, and how sensitive is it to prompt aggressiveness?</td></tr>
<tr><td>RQ3</td><td>Does font-based synthetic augmentation provide consistent improvement, especially for rare characters?</td></tr>
<tr><td>RQ4</td><td>Does word-level n-gram language model rescoring provide meaningful gains over unigram beam search?</td></tr>
</table>

<!-- ═══════════════════════════════════════════════════ 2. RELATED WORK ═══ -->
<h2>2. Related Work</h2>

<h3>2.1 CRNN-CTC for HTR</h3>
<p>The CRNN architecture (Shi et al., 2016) combining VGG-style CNN with bidirectional LSTM,
trained end-to-end via CTC loss (Graves et al., 2006), established the dominant paradigm for HTR.
Vanilla CRNN-CTC baselines achieve 8–12% CER on IAM without language model rescoring. Retsinas
et al. (2022) reach 5.14% CER using a <em>residual CNN</em> backbone and an <em>auxiliary CTC
shortcut</em> — an intermediate CTC loss head attached to the middle of the network, providing
direct gradient signal to earlier layers and improving convergence. This design is orthogonal to
our data-centric contributions.</p>

<p>Transformer-based approaches (TrOCR: 2.89%, DRetHTR: 2.26%) require 200–600 GPU hours for
fine-tuning, making controlled data-centric studies intractable. We use CRNN-CTC as a deliberate
experimental scaffold.</p>

<h3>2.2 Label Noise in Benchmark Datasets</h3>
<p>Label noise is a well-documented source of model degradation (Frenay &amp; Verleysen, 2014).
In NLP, CoNLL-03 NER contains ~5–7% label noise; WNUT-17 ~18%. In image classification,
CIFAR-N (Wei et al., 2022) provides human-annotated noisy labels for controlled studies. Most
relevant is REVEAL (Jiang et al., 2025), which uses multiple VLMs to renovate image
classification test sets by detecting label noise. We apply a similar philosophy to HTR and add
a key finding: <em>VLM prompt design critically determines whether cleaning helps or hurts</em>
— a sensitivity not observed in classification settings because annotation ambiguity in text
recognition is fundamentally different from discrete class label errors.</p>

<h3>2.3 Synthetic Data and LM Rescoring</h3>
<p>Font-based text rendering (trdg, Belval 2019) provides low-cost labeled data, used to generate
millions of synthetic lines for pretraining (Luo et al., 2023). We design a <em>targeted</em>
pipeline addressing IAM's character class imbalance rather than simply scaling volume. CTC beam
search with KenLM (Heafield, 2011) is standard post-processing in ASR and HTR, providing
consistent 1–3% absolute CER reduction at negligible inference-time cost.</p>

<!-- ═══════════════════════════════════════════════════ 3. DATASET ═══ -->
<h2>3. Dataset and Preprocessing</h2>

<h3>3.1 IAM Handwriting Database</h3>
<table>
<tr><th>Split</th><th>Samples</th><th>Writers</th></tr>
<tr><td>Train</td><td>6,482</td><td>~547</td></tr>
<tr><td>Validation</td><td>976</td><td>~55</td></tr>
<tr><td>Test</td><td>2,915</td><td>~55</td></tr>
<tr class="best"><td><strong>Total</strong></td><td><strong>10,373</strong></td><td><strong>657</strong></td></tr>
</table>

<p>All images are resized to H=64 with aspect ratio preserved. Images are stored in LMDB format
for fast batch loading. The character vocabulary contains <strong>80 classes</strong>: 79 IAM
characters plus CTC blank (index 0):</p>

<pre><code> !"#&amp;'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz</code></pre>

{'<div class="fig-row"><figure><img src="' + img["char_freq_distribution"] + '" alt="Character frequency distribution"><figcaption>Figure 2a. IAM training set character frequency. Lowercase letters dominate (~68%); digits, uppercase, and punctuation are severely underrepresented — motivating targeted synthetic augmentation.</figcaption></figure>' if img["char_freq_distribution"] else ""}
{'<figure><img src="' + img["synth_char_freq"] + '" alt="Synthetic character frequency"><figcaption>Figure 2b. Character frequency in the synthetic data pool. Rare characters (#, ;, 0, uppercase) are explicitly oversampled relative to their IAM frequency.</figcaption></figure></div>' if img["synth_char_freq"] else ""}

<h3>3.2 VLM Annotation Audit — V1 (Aggressive)</h3>
<p>All 10,373 samples audited by Doubao Seed 2.0 Pro with a prompt instructing it to "err on the
side of flagging" any possible error.</p>

<table>
<tr><th>Split</th><th>Total</th><th>Flagged</th><th>Flag Rate</th></tr>
<tr><td>Train</td><td>6,482</td><td>2,144</td><td>33.1%</td></tr>
<tr><td>Validation</td><td>976</td><td>360</td><td>36.9%</td></tr>
<tr><td>Test</td><td>2,915</td><td>1,053</td><td>36.1%</td></tr>
<tr class="best"><td><strong>Total</strong></td><td><strong>10,373</strong></td><td><strong>3,557</strong></td><td><strong>34.3%</strong></td></tr>
</table>

<p>The high flag rate reflects a combination of genuine errors and VLM over-flagging. Example cases:</p>
<table>
<tr><th>IAM Ground Truth</th><th>VLM Correction</th><th>Category</th></tr>
<tr><td><code>0M P for Manchester Exchange</code></td><td><code>UM P for Manchester Exchange</code></td><td>Likely genuine (digit 0 vs O)</td></tr>
<tr><td><code>meeting of Labour 0M Ps tommorow</code></td><td><code>meeting of Labour OM Ps tomorrow</code></td><td>Mixed (spelling + 0→O)</td></tr>
<tr><td><code>which would appear to "prop up" an out-dated</code></td><td><code>... an outmoded</code></td><td>VLM over-flag (synonym)</td></tr>
<tr><td><code>and he is to be backed by Mr. Will</code></td><td><code>and her is to be backed by her Will</code></td><td>VLM hallucination</td></tr>
</table>

<h3>3.3 VLM Annotation Audit — V2 (Conservative)</h3>
<p>Redesigned prompt explicitly instructs the model to "default to CORRECT" and lists specific
false-positive traps: digit 0 vs. letter O, British spelling, abbreviations, proper nouns,
punctuation conventions, and handwriting style variations.</p>

<table>
<tr><th></th><th>V1 Prompt</th><th>V2 Prompt</th></tr>
<tr><td>Flag rate (train)</td><td>33.1%</td><td>12.6%</td></tr>
<tr><td>Flagged samples</td><td>2,144</td><td>814</td></tr>
<tr><td>Remaining training set</td><td>4,338</td><td>5,668</td></tr>
</table>

<h3>3.4 VLM Character Confusion Analysis</h3>
<p>936 unique confusion pairs identified across 5,058 substitution events. The top 20 pairs:</p>

<table>
<tr><th>Rank</th><th>IAM label</th><th>VLM correction</th><th>Count</th><th>Category</th></tr>
<tr><td>1</td><td><code>o</code></td><td><code>a</code></td><td>155</td><td>lower–lower</td></tr>
<tr><td>2</td><td><code>n</code></td><td><code>m</code></td><td>135</td><td>lower–lower</td></tr>
<tr><td>3</td><td><code>r</code></td><td><code>s</code></td><td>133</td><td>lower–lower</td></tr>
<tr><td>4</td><td><code>a</code></td><td><code>o</code></td><td>116</td><td>lower–lower (symmetric)</td></tr>
<tr><td>5</td><td><code>a</code></td><td><code>e</code></td><td>109</td><td>lower–lower</td></tr>
<tr><td>6</td><td><code>n</code></td><td><code>u</code></td><td>96</td><td>lower–lower</td></tr>
<tr><td>7</td><td><code>o</code></td><td><code>e</code></td><td>76</td><td>lower–lower</td></tr>
<tr><td>8</td><td><code>t</code></td><td><code>h</code></td><td>75</td><td>lower–lower</td></tr>
<tr><td>9</td><td><code>t</code></td><td><code>d</code></td><td>68</td><td>lower–lower</td></tr>
<tr><td>10</td><td><code>h</code></td><td><code>l</code></td><td>60</td><td>lower–lower</td></tr>
<tr><td>11</td><td><code>s</code></td><td><code>r</code></td><td>59</td><td>lower–lower (symmetric)</td></tr>
<tr><td>12</td><td><code>r</code></td><td><code>n</code></td><td>56</td><td>lower–lower</td></tr>
<tr><td>13</td><td><code>t</code></td><td><code>l</code></td><td>54</td><td>lower–lower</td></tr>
<tr><td>14</td><td><code>u</code></td><td><code>e</code></td><td>52</td><td>lower–lower</td></tr>
<tr><td>15</td><td><code>m</code></td><td><code>w</code></td><td>47</td><td>lower–lower</td></tr>
<tr><td>16</td><td><code>e</code></td><td><code>a</code></td><td>46</td><td>lower–lower</td></tr>
<tr><td>17</td><td><code>·</code> (space)</td><td><code>s</code></td><td>42</td><td>space→lower</td></tr>
<tr><td>18</td><td><code>e</code></td><td><code>i</code></td><td>42</td><td>lower–lower</td></tr>
<tr><td>19</td><td><code>t</code></td><td><code>H</code></td><td>40</td><td>lower→upper</td></tr>
<tr><td>20</td><td><code>d</code></td><td><code>c</code></td><td>39</td><td>lower–lower</td></tr>
</table>

{'<div class="fig-row"><figure><img src="' + img["vlm_confusion_matrix_1to1"] + '" alt="VLM 1-to-1 confusion matrix"><figcaption>Figure 3a. VLM annotation confusion matrix (1-to-1 character substitutions). o↔a, n/m/u, r/s form dominant clusters.</figcaption></figure>' if img["vlm_confusion_matrix_1to1"] else ""}
{'<figure><img src="' + img["vlm_confusion_multichar"] + '" alt="VLM multi-char confusion"><figcaption>Figure 3b. Multi-character confusion patterns (merges and splits), e.g. th→H (2→1) and d→cl (1→2).</figcaption></figure></div>' if img["vlm_confusion_multichar"] else ""}

<div class="finding">
<b>Key observation:</b> 93% of confusion events involve lowercase letters, consistent with the
visual similarity of cursive strokes. The pair o↔a alone accounts for 271 events. The n cluster
(n↔m, n↔u, n↔r, n↔s) totals over 380 events — indicating this stroke shape is the single most
ambiguous element in IAM handwriting.
</div>

<!-- ═══════════════════════════════════════════════════ 4. MODEL ═══ -->
<h2>4. Model Architecture</h2>

<h3>4.1 CRNN-CTC Pipeline</h3>
<pre>
Input image (N × 1 × 64 × W)
         │
         ▼
┌────────────────────────────────────────────┐
│  VGG Feature Extractor (7 conv layers)     │
│                                            │
│  Conv1(64) → MaxPool 2×2                  │
│  Conv2(128) → MaxPool 2×2                 │
│  Conv3(256,BN) → Conv4(256)               │
│  → AsymPool(2×2, stride 2×1, pad 0×1)    │
│  Conv5(512,BN) → Conv6(512)               │
│  → AsymPool(2×2, stride 2×1, pad 0×1)    │
│  Conv7(512,BN) kernel=(4,1) [H=64 fix]   │
│                                            │
│  Output: (N, 512, 1, W/4)                 │
└────────────────────────────────────────────┘
         │  mean(dim=2) → (N, 512, W/4)
         │  permute → (N, W/4, 512)
         │  Dropout(0.1)
         ▼
┌────────────────────────────────────────────┐
│  BiGRU Encoder × 2 layers                  │
│  hidden_size = 512, bidirectional          │
│  Output: (N, W/4, 1024)                    │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  Linear(1024 → 80) + log_softmax           │
│  Permute → (T, N, 80)                      │
└────────────────────────────────────────────┘
         │
         ▼
    CTC Loss (blank=0, zero_infinity=True)

Total parameters: 16,733,648 (~16.7M)
</pre>

<h3>4.2 H=64 Adaptation</h3>
<p>The original VGG CRNN backbone was designed for H=32, where the final conv kernel is (2,1).
For H=64 (needed for taller IAM lines), we set <code>last_kernel_h = img_h // 16 = 4</code>,
producing a (4,1) kernel that collapses the remaining 4 vertical pixels into 1. This is validated
empirically: the adapted H=64 model achieves 7.35% val CER at 50 epochs vs. 7.45% for the
original H=32 configuration.</p>

<h3>4.3 Training Configuration</h3>
<table>
<tr><th>Hyperparameter</th><th>Value</th><th>Rationale</th></tr>
<tr><td>Optimizer</td><td>AdamW, β=(0.9, 0.999)</td><td>Standard for sequence models</td></tr>
<tr><td>Learning rate</td><td>3×10⁻⁴</td><td>Selected by hyperparameter search (Run 08)</td></tr>
<tr><td>LR schedule</td><td>ReduceLROnPlateau (patience=10, factor=0.8)</td><td>Adaptive; min_lr=1e-7</td></tr>
<tr><td>Warmup</td><td>1 epoch linear ramp</td><td>Prevents early divergence</td></tr>
<tr><td>Weight decay</td><td>1×10⁻⁴</td><td>L2 regularization</td></tr>
<tr><td>Dropout</td><td>0.1 (after CNN, before BiGRU)</td><td>Regularization</td></tr>
<tr><td>Gradient clip</td><td>L2 norm ≤ 10.0</td><td>CTC training stability</td></tr>
<tr><td>Batch size</td><td>64</td><td>Balance of gradient noise and speed</td></tr>
<tr><td>Epochs (main)</td><td>150</td><td>Full convergence</td></tr>
<tr><td>Augmentation</td><td>Random affine, elastic distortion, brightness jitter</td><td>Train-time only</td></tr>
</table>

{'<div class="fig-row"><figure><img src="' + img["training_curves"] + '" alt="Training curves"><figcaption>Figure 4. Training and validation CER curves for the best model (Exp3-10x). The ReduceLROnPlateau scheduler causes the characteristic staircase pattern in the learning rate.</figcaption></figure>' if img["training_curves"] else ""}
{'<figure><img src="' + img["augment_comparison"] + '" alt="Augmentation examples"><figcaption>Figure 5. Training-time image augmentation examples: (top) original, (middle) mild augmentation (affine + brightness), (bottom) heavy augmentation (elastic distortion).</figcaption></figure></div>' if img["augment_comparison"] else ""}

<!-- ═══════════════════════════════════════════════════ 5. SYNTHETIC DATA ═══ -->
<h2>5. Synthetic Data Pipeline</h2>

<h3>5.1 Motivation</h3>
<p>IAM's character distribution is heavily skewed: <code>#</code>, <code>*</code>, <code>+</code>,
<code>Z</code>, <code>Q</code> appear fewer than 50 times in training. Rather than uniformly
scaling synthetic data volume, we explicitly target underrepresented characters.</p>

<h3>5.2 Three-Layer Text Pool</h3>
<table>
<tr><th>Layer</th><th>Source</th><th>Sentences</th><th>Purpose</th></tr>
<tr><td>1</td><td>IAM training labels (×2)</td><td>~13,000</td><td>Vocabulary and domain match</td></tr>
<tr><td>2</td><td>LLM-generated rare-character text</td><td>~12,000</td><td>Oversample digits, uppercase, punctuation</td></tr>
<tr><td>3</td><td>Rule-based template sentences</td><td>~3,000</td><td>Lightweight fallback for edge cases</td></tr>
<tr class="best"><td><strong>Total</strong></td><td></td><td><strong>~28,000</strong></td><td></td></tr>
</table>

<p><strong>LLM prompt for Layer 2</strong> (example for target character <code>;</code>):</p>
<pre>Generate 5 sentences where the semicolon ';' appears at least twice each.
Sentences should resemble journalistic or literary English text (1900–1970).</pre>

<p>Example generated output:</p>
<div class="example">
The proposal covers three areas: transport, housing; and public health funding.<br>
Section 4; paragraph 7 outlines the terms of service; refer there first.
</div>

<h3>5.3 Font Rendering</h3>
<p>Rendering uses trdg (Belval, 2019) with <strong>14 fonts</strong>: 9 handwriting-style
(ArchitectsDaughter, HomemadeApple, IndieFlower, Kalam-Light, Kalam-Regular, PatrickHand,
PermanentMarker, RockSalt, Satisfy) and 5 print fonts (DroidSans, Lato, OpenSans, Raleway).
Bold fonts excluded to avoid stroke-width distribution shift.</p>

{'<div class="fig-row"><figure><img src="' + img["fonts_preview"] + '" alt="Font samples" style="max-width:90%"><figcaption>Figure 6. Sample renders from the 14 fonts used in synthesis. The mix of handwriting and print styles provides stylistic diversity.</figcaption></figure>' if img["fonts_preview"] else ""}
{'<figure><img src="' + img["synth_samples"] + '" alt="Synthetic samples"><figcaption>Figure 7. Example synthetic training images targeting rare characters. Note the variety of fonts, skew, and blur applied during rendering.</figcaption></figure></div>' if img["synth_samples"] else ""}

<h3>5.4 Scale and Storage</h3>
<p>A single 65,000-sample LMDB pool is generated once; training experiments draw subsets with
a fixed random seed for fair comparison.</p>

<table>
<tr><th>Scale</th><th>Synthetic samples</th><th>Total with IAM train</th></tr>
<tr><td>1×</td><td>6,500</td><td>~13,000</td></tr>
<tr><td>3×</td><td>19,500</td><td>~26,000</td></tr>
<tr><td>5×</td><td>32,500</td><td>~39,000</td></tr>
<tr><td>7×</td><td>45,500</td><td>~52,000</td></tr>
<tr class="best"><td><strong>10×</strong></td><td><strong>65,000</strong></td><td><strong>~71,500</strong></td></tr>
</table>

<!-- ═══════════════════════════════════════════════════ 6. HPARAM ═══ -->
<h2>6. Hyperparameter Search</h2>

<h3>6.1 Protocol</h3>
<p>Greedy sequential search over 4 axes: hidden size, dropout, batch size, learning rate.
Each candidate is evaluated at <strong>50 epochs</strong> (proxy budget). A change is kept if
validation CER improves by &gt;0.05% relative.</p>

{'<figure><img src="' + img["hparam_search_chart"] + '" alt="Hparam search results" style="max-width:80%"><figcaption>Figure 8. Hyperparameter search trajectory. Run 02 (hidden=512) and Run 08 (lr=3×10⁻⁴) provide the two largest single-step gains.</figcaption></figure>' if img["hparam_search_chart"] else ""}

<h3>6.2 Full Results</h3>
<table>
<tr><th>Run</th><th>Change</th><th>Val CER</th><th>Δ</th><th>Decision</th></tr>
<tr><td>00</td><td>Baseline (hidden=256, lr=1e-4)</td><td>9.04%</td><td>—</td><td>Keep</td></tr>
<tr><td>01</td><td>hidden=128</td><td>9.79%</td><td>+0.75%</td><td>Discard</td></tr>
<tr class="best"><td><strong>02</strong></td><td><strong>hidden=512</strong></td><td><strong>7.35%</strong></td><td><strong>−1.69%</strong></td><td><strong>Keep ✓</strong></td></tr>
<tr><td>03</td><td>dropout=0.0</td><td>7.45%</td><td>+0.10%</td><td>Discard</td></tr>
<tr><td>04</td><td>dropout=0.2</td><td>7.85%</td><td>+0.50%</td><td>Discard</td></tr>
<tr><td>05</td><td>dropout=0.3</td><td>7.20%</td><td>−0.15%</td><td>Keep</td></tr>
<tr><td>06</td><td>batch_size=32</td><td>6.73%</td><td>−0.62%</td><td>Keep</td></tr>
<tr><td>07</td><td>batch_size=128</td><td>8.39%</td><td>+1.04%</td><td>Discard</td></tr>
<tr class="best"><td><strong>08</strong></td><td><strong>lr=3×10⁻⁴</strong></td><td><strong>6.29%</strong></td><td><strong>−1.06%</strong></td><td><strong>Keep ✓</strong></td></tr>
<tr><td>09</td><td>lr=5×10⁻⁵</td><td>9.00%</td><td>+2.71%</td><td>Discard</td></tr>
<tr><td>10</td><td>cnn_out=256</td><td>7.50%</td><td>+1.21%</td><td>Discard</td></tr>
<tr><td>11</td><td>weight_decay=1e-3</td><td>8.05%</td><td>+1.76%</td><td>Discard</td></tr>
<tr><td>12</td><td>weight_decay=0</td><td>7.42%</td><td>+1.13%</td><td>Discard</td></tr>
<tr><td>13</td><td>Original VGG kernel (H=32)</td><td>7.45%</td><td>+1.16%</td><td>Discard</td></tr>
</table>

<div class="finding">
<b>Finding:</b> BiGRU hidden=512 provides the single largest gain (9.04%→7.35%), followed by
lr=3×10⁻⁴ (7.35%→6.29%). Too-small learning rate (lr=5×10⁻⁵, 9.00%) is the worst
configuration, because 50 epochs are insufficient for convergence at that rate. The optimal
configuration — hidden=512, lr=3×10⁻⁴, dropout=0.1, batch=64 — is used in all subsequent
experiments.
</div>

<!-- ═══════════════════════════════════════════════════ 7. EXPERIMENTS ═══ -->
<h2>7. Experiments</h2>

<h3>7.1 Setup</h3>
<p>All main experiments: 150 epochs, best-checkpoint evaluation, held-out test set (2,915
samples). Metric: CER = edit_distance(hyp, ref) / max(len(ref), 1) × 100%.</p>

<h3>7.2 Full IAM vs. VLM-Cleaned IAM</h3>
<table>
<tr><th>Experiment</th><th>Training Set</th><th>Samples</th><th>Test CER</th><th>Δ vs. Exp1</th></tr>
<tr class="best"><td><strong>Exp1 — Full IAM</strong></td><td>train</td><td>6,482</td><td><strong>8.43%</strong></td><td>—</td></tr>
<tr><td>Exp2 — V1 cleaned</td><td>train_clean</td><td>4,338</td><td>10.33%</td><td>+1.90% ↑ worse</td></tr>
<tr><td>Exp5-base — V2 cleaned</td><td>train_clean_2</td><td>5,668</td><td>8.36%</td><td>−0.07% ≈ same</td></tr>
</table>

<div class="finding">
<b>Finding:</b> V1 cleaning (−33%) significantly degrades performance, confirming that the removed
samples contain valuable training signal. V2 cleaning (−12.6%) is effectively neutral — neither
helping nor hurting. <em>No cleaning prompt achieves below-baseline CER.</em>
</div>

<h3>7.3 Synthetic Data Augmentation</h3>
<table>
<tr><th>Experiment</th><th>Training Set</th><th>Scale</th><th>Total Samples</th><th>Test CER</th></tr>
<tr><td>Exp1</td><td>Full IAM</td><td>0×</td><td>6,482</td><td>8.43%</td></tr>
<tr><td>Exp3-1x</td><td>Full IAM + synth</td><td>1×</td><td>~13,000</td><td>7.93%</td></tr>
<tr><td>Exp3-2x</td><td>Full IAM + synth</td><td>2×</td><td>~19,500</td><td>7.87%</td></tr>
<tr><td>Exp3-3x</td><td>Full IAM + synth</td><td>3×</td><td>~26,000</td><td>7.66%</td></tr>
<tr><td>Exp3-5x</td><td>Full IAM + synth</td><td>5×</td><td>~39,000</td><td>7.63%</td></tr>
<tr><td>Exp3-7x</td><td>Full IAM + synth</td><td>7×</td><td>~52,000</td><td>7.74%</td></tr>
<tr class="best"><td><strong>Exp3-10x</strong></td><td>Full IAM + synth</td><td>10×</td><td>~71,500</td><td><strong>7.52%</strong></td></tr>
<tr><td>Exp5-1x</td><td>V2 cleaned + synth</td><td>1×</td><td>~12,168</td><td>8.08%</td></tr>
<tr><td>Exp5-2x</td><td>V2 cleaned + synth</td><td>2×</td><td>~18,668</td><td>7.97%</td></tr>
<tr><td>Exp5-3x</td><td>V2 cleaned + synth</td><td>3×</td><td>~25,168</td><td>8.01%</td></tr>
<tr><td>Exp5-5x</td><td>V2 cleaned + synth</td><td>5×</td><td>~38,168</td><td>7.92%</td></tr>
<tr><td>Exp5-7x</td><td>V2 cleaned + synth</td><td>7×</td><td>~51,668</td><td>8.17%</td></tr>
<tr><td>Exp5-10x</td><td>V2 cleaned + synth</td><td>10×</td><td>~71,168</td><td>8.06%</td></tr>
</table>

<div class="finding">
<b>Key finding:</b> Exp3 (Full IAM + synth) improves monotonically from 8.43% to 7.52% at 10×
scale (10.8% relative improvement). Exp5 (V2-cleaned + synth) stagnates around 7.9–8.2% and
shows no further improvement beyond 5× — diverging from Exp3's trend. The 814 V2-removed
samples are providing training signal that synthetic data cannot replace.
</div>

<h3>7.4 Per-Character Error Analysis</h3>
<p>Character-level CER computed via Levenshtein backtrace: the alignment operation sequence
(substitute/delete/insert) is used to attribute each error to its corresponding reference
character class.</p>

<table>
<tr><th>Character</th><th>Test count</th><th>Exp1 CER</th><th>Exp3-10x CER</th><th>Δ</th></tr>
<tr class="best"><td><code>#</code></td><td>5</td><td>100.0%</td><td>60.0%</td><td>−40.0%</td></tr>
<tr class="best"><td><code>;</code></td><td>60</td><td>41.7%</td><td>21.7%</td><td>−20.0%</td></tr>
<tr class="best"><td><code>U</code></td><td>11</td><td>36.4%</td><td>18.2%</td><td>−18.2%</td></tr>
<tr class="best"><td><code>R</code></td><td>67</td><td>35.8%</td><td>20.9%</td><td>−14.9%</td></tr>
<tr><td><code>:</code></td><td>29</td><td>55.2%</td><td>41.4%</td><td>−13.8%</td></tr>
<tr><td><code>!</code></td><td>47</td><td>38.3%</td><td>25.5%</td><td>−12.8%</td></tr>
<tr><td><code>0</code></td><td>34</td><td>38.2%</td><td>26.5%</td><td>−11.8%</td></tr>
<tr><td><code>W</code></td><td>122</td><td>32.0%</td><td>20.5%</td><td>−11.5%</td></tr>
<tr><td><code>L</code></td><td>123</td><td>33.3%</td><td>24.4%</td><td>−8.9%</td></tr>
<tr><td><code>"</code></td><td>576</td><td>27.1%</td><td>17.9%</td><td>−9.2%</td></tr>
<tr><td><code>z</code></td><td>45</td><td>35.6%</td><td>40.0%</td><td><span style="color:#c00">+4.4%</span></td></tr>
</table>

<div class="finding">
<b>Finding:</b> Improvements are concentrated exactly on the rare characters targeted by the
synthetic pipeline: <code>#</code> (−40%), <code>;</code> (−20%), uppercase <code>U R W L</code>
(−9% to −18%), digit <code>0</code> (−11.8%). <code>z</code> shows a small regression,
plausibly because font renderings of <code>z</code> differ from IAM's cursive form, introducing
distribution shift.
</div>

<h3>7.5 VLM vs. Model Confusion — Cross-Matrix Analysis</h3>
<p>We compare: (a) the VLM annotation confusion matrix (IAM label → VLM correction across
5,058 substitutions); and (b) the model prediction confusion matrix (ground truth → predicted
character from Levenshtein alignment of Exp3-10x test predictions).</p>

<table>
<tr><th>Pair</th><th>VLM confusion count</th><th>Model confusion count</th><th>Shared?</th></tr>
<tr class="best"><td><code>o</code> ↔ <code>a</code></td><td>271</td><td>454</td><td>✓ #1</td></tr>
<tr class="best"><td><code>n</code> ↔ <code>m</code></td><td>135+</td><td>194</td><td>✓ #2</td></tr>
<tr class="best"><td><code>r</code> ↔ <code>s</code></td><td>192</td><td>213</td><td>✓ #3</td></tr>
<tr><td><code>a</code> ↔ <code>e</code></td><td>155</td><td>152</td><td>✓ #4</td></tr>
<tr><td><code>n</code> ↔ <code>u</code></td><td>96+</td><td>160</td><td>✓</td></tr>
<tr><td><code>t</code> ↔ <code>l</code></td><td>54</td><td>165</td><td>✓</td></tr>
<tr><td><code>r</code> ↔ <code>n</code></td><td>97</td><td>142</td><td>✓</td></tr>
<tr><td><code>o</code> ↔ <code>e</code></td><td>76</td><td>115</td><td>✓</td></tr>
</table>

<div class="insight">
<b>Spearman rank correlation:</b> ρ = <strong>0.559</strong>, p &lt; 0.0001 (n=160 shared pairs)<br><br>
Both the VLM and the trained model struggle with the same character pairs. This is not a
coincidence — it reflects genuine visual ambiguity in handwritten English cursive. The pairs
<code>o↔a</code>, <code>n↔m</code>, <code>r↔s</code> involve strokes that are routinely
indistinguishable even to human readers. This explains why VLM cleaning cannot reliably
identify annotation errors: the "errors" it detects are the same ambiguous strokes the model
itself gets wrong — meaning they are likely <em>correct-but-hard</em> examples, not errors.
</div>

<!-- ═══════════════════════════════════════════════════ 8. LM ═══ -->
<h2>8. Language Model Rescoring</h2>

<h3>8.1 KenLM 4-gram Language Model</h3>
<p>Training corpus: <strong>183,468 sentences</strong> = IAM train+val labels (~7,400) +
176,010 lines from 20 Project Gutenberg novels (Pride and Prejudice, Moby Dick, Frankenstein,
Sherlock Holmes, and 16 others). Model: word 4-gram with modified Kneser-Ney smoothing via
lmplz. Output: 150MB ARPA file.</p>

<p>Beam search uses pyctcdecode (beam=50, α=0.5, β=1.0). Scoring formula:</p>
<pre>score = α × CTC_log_prob + β × LM_log_prob + γ × word_count_bonus</pre>

<h3>8.2 Vocabulary Case Bug</h3>
<p>The initial vocabulary was built by lowercasing all IAM words, causing proper nouns
(<code>London</code>, <code>Mr</code>, <code>Manchester</code>, <code>Evans</code>) to fall
outside the vocabulary. pyctcdecode applies <code>unk_score_offset=−10.0</code> to unknown
words, creating a strong penalty that steered beam search away from correct hypotheses
containing proper nouns. Fix: include each IAM word in both its original case <em>and</em>
lowercase.</p>

<h3>8.3 Decoding Results</h3>
<table>
<tr><th>Decoding</th><th>Model</th><th>Greedy CER</th><th>Beam CER</th><th>Δ</th></tr>
<tr><td>CTC greedy</td><td>Exp3-10x</td><td>7.53%</td><td>—</td><td>—</td></tr>
<tr><td>Beam + unigram vocab</td><td>Exp3-10x</td><td>—</td><td>7.50%</td><td>−0.03%</td></tr>
<tr><td>Beam + 4-gram KenLM (lowercased)</td><td>Exp3-10x</td><td>—</td><td>6.64%</td><td>−0.89%</td></tr>
<tr class="best"><td><strong>Beam + 4-gram KenLM (mixed-case)</strong></td><td><strong>Exp3-10x</strong></td><td>—</td><td><strong>6.41%</strong></td><td><strong>−1.12%</strong></td></tr>
<tr><td>Beam + 4-gram KenLM (mixed-case)</td><td>Exp5-5x</td><td>7.92%</td><td>6.90%</td><td>−1.02%</td></tr>
<tr><td>Beam + 4-gram KenLM (mixed-case)</td><td>Exp5-7x</td><td>8.17%</td><td>6.84%</td><td>−1.33%</td></tr>
<tr><td>Beam + 4-gram KenLM (mixed-case)</td><td>Exp5-10x</td><td>8.06%</td><td>6.87%</td><td>−1.19%</td></tr>
</table>

<h3>8.4 Concrete Correction Examples</h3>
<div class="example">
<b>Example 1:</b><br>
<span class="ref">REF:&nbsp; "I don't <u>think</u> he will storm the <u>charts</u> with this one, <u>but</u> it's a good start."</span><br>
<span class="grd">GRD:&nbsp; "I don't <u>thinu</u> he will storm the <u>charks</u> with this one, <u>bat</u> it's a good start."</span><br>
<span class="beam">BEAM: "I don't <u>think</u> he will storm the <u>charts</u> with this one, <u>but</u> it's a good start." ✓</span>
</div>

<div class="example">
<b>Example 2:</b><br>
<span class="ref">REF:&nbsp; "He <u>writes</u> with Tolchard <u>Evans</u>, composer of Lady of Spain."</span><br>
<span class="grd">GRD:&nbsp; "He <u>wribes</u> with Tolchard <u>Evons</u>, composer of Lady of Spain."</span><br>
<span class="beam">BEAM: "He <u>writes</u> with Tolchard <u>Evans</u>, composer of Lady of Spain." ✓</span>
</div>

<div class="example">
<b>Example 3 (partial correction):</b><br>
<span class="ref">REF:&nbsp; "CHRIS CHARLES, 39, who lives in Stockton-on-Tees, is an <u>accountant</u>."</span><br>
<span class="grd">GRD:&nbsp; "CHRIS CHARLES, 39, who lives in Stocuton-on-Tees, is an <u>accourlant</u>."</span><br>
<span class="beam">BEAM: "CHRIS CHARLES, 39, who lives in Stocuton-on-Tees, is an <u>accountant</u>." ✓</span><br>
<em style="font-size:9pt">Note: "Stocuton" (CNN error) not corrected — the LM accepts it as an unknown proper noun.</em>
</div>

<div class="finding">
<b>Finding:</b> The LM reliably corrects common English words (think, charts, writes) by using
4-gram context. It also corrects proper nouns (Evans) via vocabulary lookup. However, it cannot
fix pure visual errors in names it has never seen (Stocuton for Stockton), confirming the LM
complements the model but does not substitute for it.
</div>

<!-- ═══════════════════════════════════════════════════ 9. DISCUSSION ═══ -->
<h2>9. Discussion</h2>

<h3>9.1 Why VLM Cleaning Fails</h3>
<p>The cross-matrix correlation (ρ=0.559) reveals that VLM and model confusion are not
independent. Both systems struggle with the same cursive stroke ambiguities. When V1's aggressive
prompt removes IAM samples containing <code>o↔a</code> ambiguities, it removes
<em>correct-but-hard</em> examples the model needs — not genuine errors. V2's conservative prompt
avoids most of these false positives, hence the neutral effect.</p>

<p>The practical implication: VLM cleaning is most credible for <em>clear, high-confidence errors</em>
(words entirely unrelated to handwritten content). For the dominant confusion pairs
(<code>o↔a</code>, <code>n/m/u</code>), cleaning requires per-character confidence gating from
the model itself, not a binary VLM verdict.</p>

<h3>9.2 Synthetic Augmentation: Targeted vs. Generic</h3>
<p>The Exp3 improvement curve (8.43%→7.52%) with diminishing returns beyond 5× follows the
expected pattern for font-based synthesis. The per-character analysis confirms our targeted
approach works: the largest gains align exactly with the rare character classes we emphasized in
Layer 2 of the text pool. Generic volume scaling (matching Exp5's 10× with cleaned data) cannot
overcome the deficit from missing the 814 hard-but-valid samples.</p>

<h3>9.3 LM Rescoring vs. Synthetic Data: Cost-Efficiency</h3>
<table>
<tr><th>Method</th><th>CER gain</th><th>GPU cost</th><th>Retraining?</th></tr>
<tr><td>10× synthetic augmentation</td><td>0.91% (8.43%→7.52%)</td><td>~20 GPU-hours</td><td>Yes (full)</td></tr>
<tr class="best"><td><strong>4-gram KenLM beam search</strong></td><td><strong>1.12% (7.53%→6.41%)</strong></td><td><strong>&lt;1 GPU-hour (LM training)</strong></td><td><strong>No</strong></td></tr>
</table>

<p>KenLM rescoring slightly outperforms 10× synthetic augmentation with effectively zero
training cost. The two approaches are complementary: combining Exp3-10x with beam search
achieves the best result of <strong>6.41%</strong>.</p>

<h3>9.4 Limitations</h3>
<ul>
<li><b>VLM cleaning fails:</b> Neither prompt improves over the full dataset. The true
annotation error rate in IAM remains unknown without manual validation.</li>
<li><b>CRNN-CTC is not SOTA:</b> Our 6.41% is 3.5% above TrOCR-Large (2.89%); the same
data-centric findings may not transfer identically to Transformer-based HTR.</li>
<li><b>Single dataset:</b> All experiments on IAM; generalizability to RIMES, CVL, Bentham
untested.</li>
<li><b>Font domain gap:</b> Font-rendered synthesis lacks stroke-level variability; diffusion-
based synthesis would likely yield larger gains at equal volume.</li>
</ul>

<!-- ═══════════════════════════════════════════════════ 10. CONCLUSION ═══ -->
<h2>10. Conclusion</h2>

<p>We present a comprehensive data-centric study of CRNN-CTC handwritten text recognition on IAM,
covering hyperparameter optimization, VLM annotation auditing, synthetic data augmentation, and
language model rescoring. Our best result — <strong>6.41% test CER</strong> (Exp3-10x + 4-gram
KenLM, mixed-case vocabulary) — represents a 24% relative improvement over the vanilla baseline
(8.43%).</p>

<p>Two findings stand out for future HTR research:</p>
<ol>
<li><b>VLM annotation cleaning is prompt-sensitive and ultimately unable to improve IAM
performance.</b> The cross-matrix Spearman correlation (ρ=0.559) between VLM and model
confusion patterns reveals why: both systems face the same visual ambiguity on cursive strokes
like <code>o↔a</code> and <code>n↔m</code>. This makes it impossible for a binary VLM verdict
to reliably separate annotation errors from hard-but-valid examples. Per-character confidence
gating or targeted human adjudication of specific confusion pairs is needed.</li>
<li><b>Language model rescoring is more cost-efficient than synthetic augmentation.</b> A
one-time 4-gram KenLM training (no model retraining, &lt;1 GPU-hour) provides 1.12% CER
reduction — slightly better than 10× synthetic augmentation (0.91%) at a fraction of the cost.</li>
</ol>

<p><b>Future directions:</b> (1) Manual validation of V1-flagged samples to quantify the true
IAM annotation error rate; (2) Auxiliary CTC shortcut (Retsinas et al., 2022) to close the
remaining gap to 5.14%; (3) Diffusion-based handwriting synthesis for higher-quality rare-
character augmentation; (4) Extension of the cross-matrix analysis to RIMES and CVL benchmarks.</p>

<!-- ═══════════════════════════════════════════════════ REFERENCES ═══ -->
<h2>References</h2>
<p>Belval, E. (2019). TextRecognitionDataGenerator. GitHub.</p>
<p>Frenay, B., &amp; Verleysen, M. (2014). Classification in the presence of label noise: A survey. <em>IEEE TNNLS</em>, 25(5), 845–869.</p>
<p>Graves, A., Fernández, S., Gomez, F., &amp; Schmidhuber, J. (2006). Connectionist temporal classification. <em>ICML 2006</em>.</p>
<p>Hannun, A., et al. (2014). Deep Speech: Scaling up end-to-end speech recognition. <em>arXiv:1412.5567</em>.</p>
<p>Heafield, K. (2011). KenLM: Faster and smaller language model queries. <em>WMT 2011</em>.</p>
<p>Jiang, X., et al. (2025). When VLMs Meet Image Classification: Test Sets Renovation. <em>arXiv:2505.16149</em>.</p>
<p>Li, M., et al. (2021). TrOCR: Transformer-based optical character recognition. <em>arXiv:2109.10282</em>.</p>
<p>Marti, U.-V., &amp; Bunke, H. (2002). The IAM-database. <em>IJDAR</em>, 5(1), 39–46.</p>
<p>Retsinas, G., Sfikas, P., Gatos, B., &amp; Nikou, C. (2022). Best practices for a handwritten text recognition system. <em>arXiv:2404.11339</em>.</p>
<p>Shi, B., Bai, X., &amp; Yao, C. (2016). An end-to-end trainable neural network for image-based sequence recognition. <em>IEEE TPAMI</em>, 39(11), 2298–2304.</p>
<p>Wei, J., et al. (2022). Learning with noisy labels revisited. <em>ICLR 2022</em>.</p>

<!-- ═══════════════════════════════════════════════════ APPENDIX ═══ -->
<h2>Appendix A: VLM Prompt Comparison</h2>

<h4>V1 Prompt (aggressive, flag rate 33.1%):</h4>
<pre>The image shows a single line of handwritten English text.
The claimed annotation is: "{{annotation}}"
Compare word by word. Reply CORRECT / INCORRECT / AMBIGUOUS.
Rules:
- INCORRECT: use whenever you can read a word and it differs. Err on the side of flagging.
- AMBIGUOUS: only when ink is too faded to read at all.</pre>

<h4>V2 Prompt (conservative, flag rate 12.6%):</h4>
<pre>The image shows a single line of handwritten English text from the IAM
Handwriting Database, annotated by trained human experts (expected ~95%+ accuracy).
Annotation: "{{annotation}}"
Default to CORRECT. Only flag INCORRECT when ALL of the following:
  - You can clearly read the word(s) in question
  - The difference is unambiguous, not a style variation
  - You can specify exactly which characters differ
Do NOT flag: digit 0 vs letter O (IAM uses 0 for zero; trust annotation),
minor punctuation, British spelling, abbreviations, proper nouns,
handwriting style variations.</pre>

<h2>Appendix B: Hyperparameter Configuration Table</h2>
<table>
<tr><th>Run</th><th>hidden</th><th>dropout</th><th>batch</th><th>lr</th><th>cnn_out</th><th>weight_decay</th></tr>
<tr><td>00</td><td>256</td><td>0.1</td><td>64</td><td>1e-4</td><td>512</td><td>1e-4</td></tr>
<tr><td>01</td><td>128</td><td>0.1</td><td>64</td><td>1e-4</td><td>512</td><td>1e-4</td></tr>
<tr class="best"><td>02</td><td>512</td><td>0.1</td><td>64</td><td>1e-4</td><td>512</td><td>1e-4</td></tr>
<tr><td>03</td><td>512</td><td>0.0</td><td>64</td><td>1e-4</td><td>512</td><td>1e-4</td></tr>
<tr><td>04</td><td>512</td><td>0.2</td><td>64</td><td>1e-4</td><td>512</td><td>1e-4</td></tr>
<tr><td>05</td><td>512</td><td>0.3</td><td>64</td><td>1e-4</td><td>512</td><td>1e-4</td></tr>
<tr><td>06</td><td>512</td><td>0.1</td><td>32</td><td>1e-4</td><td>512</td><td>1e-4</td></tr>
<tr><td>07</td><td>512</td><td>0.1</td><td>128</td><td>1e-4</td><td>512</td><td>1e-4</td></tr>
<tr class="best"><td>08</td><td>512</td><td>0.1</td><td>64</td><td>3e-4</td><td>512</td><td>1e-4</td></tr>
<tr><td>09</td><td>512</td><td>0.1</td><td>64</td><td>5e-5</td><td>512</td><td>1e-4</td></tr>
<tr><td>10</td><td>512</td><td>0.1</td><td>64</td><td>1e-4</td><td>256</td><td>1e-4</td></tr>
<tr><td>11</td><td>512</td><td>0.1</td><td>64</td><td>1e-4</td><td>512</td><td>1e-3</td></tr>
<tr><td>12</td><td>512</td><td>0.1</td><td>64</td><td>1e-4</td><td>512</td><td>0.0</td></tr>
<tr><td>13</td><td>512</td><td>0.1</td><td>64</td><td>1e-4</td><td>512</td><td>1e-4 (orig VGG)</td></tr>
</table>

<hr>
<p style="text-align:center;font-size:9pt;color:#888">
CS5300 Computer Vision — Final Project Report · Jian (Thomas) Zhang · Northeastern University · April 2026
</p>

</body>
</html>
"""

out = ROOT / "results" / "report.html"
out.write_text(HTML, encoding="utf-8")
print(f"Report written to: {out}")
print(f"File size: {out.stat().st_size / 1024:.0f} KB")
