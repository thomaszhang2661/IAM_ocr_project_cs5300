"""
Pixel-level character replacement using CTC forced alignment.

Pipeline per flagged sample:
  1. Run model → get CTC log-probs (T × V)
  2. Forced alignment (Viterbi) → each GT character → column range in image
  3. Diff GT vs VLM-corrected text → find substituted characters
  4. For each substitution: estimate local style (stroke width, skew)
     → render synthetic character → paste back into image
  5. Save patched image + corrected label to output LMDB

Completely independent — does NOT modify any existing training scripts.

Usage:
    python char_replace.py \
        --checkpoint checkpoints/exp1_full_iam/best.pt \
        --flagged    results/doubao_train_flagged.csv \
        --img_dir    data/iam_hf/train/images \
        --labels_csv data/iam_hf/train/labels.csv \
        --font_dir   data/handwriting_fonts \
        --output_lmdb data/lmdb/train_patched \
        --output_html results/patch_viewer.html \
        --gpu 6
"""

import argparse
import difflib
import os
import random
import sys

import cv2
import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))

IAM_ALPHABET = (
    ' !"#&\'()*+,-./0123456789:;?'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abcdefghijklmnopqrstuvwxyz'
)
BLANK = 0
char2idx = {c: i + 1 for i, c in enumerate(IAM_ALPHABET)}
idx2char = {i + 1: c for i, c in enumerate(IAM_ALPHABET)}

# CNN reduces width by this factor (4× pooling in width across layers)
CNN_WIDTH_STRIDE = 4


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(ckpt_path, device):
    from htr_model.model_v2 import build_model_v2
    model = build_model_v2(IAM_ALPHABET, hidden_size=512, dropout=0.0)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    # Strip DataParallel prefix if present
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Image preprocessing (matches LMDBDataset)
# ---------------------------------------------------------------------------
def preprocess(img_bgr, target_h=64):
    """BGR numpy → normalised float tensor (1, 1, H, W)."""
    h, w = img_bgr.shape[:2]
    new_w = max(1, int(w * target_h / h))
    img = cv2.resize(img_bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    t = torch.from_numpy(gray).float() / 255.0          # [0,1]
    t = (t - 0.5) / 0.5                                  # [-1,1]
    return t.unsqueeze(0).unsqueeze(0)                   # (1,1,H,W)


# ---------------------------------------------------------------------------
# CTC forced alignment → character column ranges
# ---------------------------------------------------------------------------
def ctc_forced_align(log_probs_T_V, label_str, device):
    """
    Returns list of (char, col_start, col_end) for each character in label_str.
    col_* are in *preprocessed image* pixel coordinates.

    log_probs_T_V: (T, V) float tensor, log-softmax output
    label_str: ground-truth string
    """
    from torchaudio.functional import forced_align, merge_tokens

    # Encode label — skip chars not in alphabet
    tokens = [char2idx[c] for c in label_str if c in char2idx]
    if not tokens:
        return []

    T = log_probs_T_V.shape[0]
    # forced_align expects (B, T, V)
    log_probs_1_T_V = log_probs_T_V.unsqueeze(0).to(device)
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)  # (1, L)
    input_lengths  = torch.tensor([T],           dtype=torch.int32, device=device)
    target_lengths = torch.tensor([len(tokens)], dtype=torch.int32, device=device)

    # forced_align returns (alignments: (B,T), scores: (B,T))
    alignments, scores = forced_align(
        log_probs_1_T_V, targets, input_lengths, target_lengths, blank=BLANK
    )
    # merge_tokens collapses repeated tokens → list of TokenSpan(token, start, end)
    spans = merge_tokens(alignments[0], scores[0], blank=BLANK)

    result = []
    for span in spans:
        char = idx2char.get(span.token, '?')
        col_start = span.start * CNN_WIDTH_STRIDE
        col_end   = (span.end + 1) * CNN_WIDTH_STRIDE
        result.append((char, col_start, col_end))

    return result


# ---------------------------------------------------------------------------
# Style estimation from a character region
# ---------------------------------------------------------------------------
def estimate_stroke_width(region_gray):
    """Estimate dominant stroke width via morphological skeleton distance."""
    if region_gray is None or region_gray.size == 0:
        return 2
    _, bw = cv2.threshold(region_gray, 128, 255, cv2.THRESH_BINARY_INV)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    val = float(np.percentile(dist[dist > 0], 75)) if dist.max() > 0 else 2.0
    return max(1.0, val)


def estimate_skew(img_gray):
    """Estimate text line skew angle in degrees."""
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
                             minLineLength=20, maxLineGap=5)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 != x1:
            angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if not angles:
        return 0.0
    angles = [a for a in angles if abs(a) < 30]
    return float(np.median(angles)) if angles else 0.0


# ---------------------------------------------------------------------------
# Synthetic character rendering
# ---------------------------------------------------------------------------
def render_synthetic_char(char, target_h, target_w, font_paths,
                           stroke_width=2.0, skew_deg=0.0):
    """
    Render a single character to a (target_h × target_w) grayscale image.
    Tries to match stroke_width and skew_deg of the surrounding text.
    """
    if not font_paths or not char:
        return None

    fp = random.choice(font_paths)
    # Binary-search font size fitting target_h
    lo, hi = 6, target_h * 2
    for _ in range(20):
        mid = (lo + hi) // 2
        try:
            f = ImageFont.truetype(fp, size=mid)
            bb = f.getbbox(char)
            if bb[3] - bb[1] <= target_h - 2:
                lo = mid
            else:
                hi = mid - 1
        except Exception:
            break
    try:
        f = ImageFont.truetype(fp, size=max(lo, 6))
    except Exception:
        return None

    bb = f.getbbox(char)
    ch = max(bb[3] - bb[1], 1)
    cw = max(bb[2] - bb[0], 1)

    # Render on large canvas, then crop
    canvas = Image.new('L', (cw + 10, target_h + 10), 255)
    draw = ImageDraw.Draw(canvas)
    y_off = (target_h - ch) // 2 - bb[1]
    draw.text((5 - bb[0], y_off), char, font=f, fill=0)

    # Apply skew via affine transform
    if abs(skew_deg) > 0.5:
        arr = np.array(canvas)
        tan = np.tan(np.radians(-skew_deg))
        h_arr, w_arr = arr.shape
        M = np.float32([[1, tan, 0], [0, 1, 0]])
        arr = cv2.warpAffine(arr, M, (w_arr + int(abs(tan) * h_arr), h_arr),
                              borderValue=255)
        canvas = Image.fromarray(arr)

    # Resize to target_w × target_h
    canvas = canvas.resize((max(target_w, 4), target_h), Image.LANCZOS)

    # Adjust stroke width via morphological ops
    arr = np.array(canvas)
    _, bw = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY_INV)
    cur_sw = estimate_stroke_width(arr)
    diff = stroke_width - cur_sw
    if diff > 0.8:
        k = max(1, int(diff))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
        bw = cv2.dilate(bw, kernel, iterations=1)
    elif diff < -0.8:
        k = max(1, int(-diff))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
        bw = cv2.erode(bw, kernel, iterations=1)
    result = 255 - bw
    return result


# ---------------------------------------------------------------------------
# Diff GT vs corrected → list of (position_in_gt, old_char, new_char)
# ---------------------------------------------------------------------------
def char_diff(gt: str, corrected: str):
    """
    Returns list of (gt_pos, old_char, new_char) for 1:1 substitutions only.
    Skips insertions / deletions (too hard to localise in image).
    """
    ops = []
    matcher = difflib.SequenceMatcher(None, gt, corrected, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace' and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                if gt[i1 + k] != corrected[j1 + k]:
                    ops.append((i1 + k, gt[i1 + k], corrected[j1 + k]))
    return ops


# ---------------------------------------------------------------------------
# Paste synthetic char into image region
# ---------------------------------------------------------------------------
def paste_char(img_gray, col_start, col_end, char_img):
    """Replace columns [col_start, col_end) in img_gray with char_img."""
    h, w = img_gray.shape
    col_start = max(0, col_start)
    col_end   = min(w, col_end)
    region_w  = col_end - col_start
    if region_w <= 0 or char_img is None:
        return img_gray

    # Resize synthetic char to region_w × h
    syn = cv2.resize(char_img, (region_w, h), interpolation=cv2.INTER_AREA)

    out = img_gray.copy()
    out[:, col_start:col_end] = syn
    return out


# ---------------------------------------------------------------------------
# Process one sample
# ---------------------------------------------------------------------------
def process_sample(img_path, gt, corrected, model, device, font_paths):
    """
    Returns (patched_img_gray, corrected_label) or None on failure.
    patched_img_gray is at H=64, original width.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    # Preprocess & forward pass
    inp = preprocess(img_bgr).to(device)                     # (1,1,64,W')
    with torch.no_grad():
        logits = model(inp)                                   # (T, 1, V)
    log_probs = F.log_softmax(logits, dim=2)[:, 0, :]        # (T, V)

    # Forced alignment on GT
    try:
        alignment = ctc_forced_align(log_probs, gt, device)
    except Exception as e:
        return None

    if len(alignment) != len(gt):
        return None

    # Compute char column ranges in ORIGINAL image (before resize)
    orig_h, orig_w = img_bgr.shape[:2]
    proc_w = inp.shape[-1]   # width after preprocess resize
    scale  = orig_w / proc_w  # map processed cols → original cols

    char_ranges = []
    for char, cs, ce in alignment:
        orig_cs = int(cs * scale)
        orig_ce = int(ce * scale)
        char_ranges.append((char, orig_cs, orig_ce))

    # Diff to find substitutions
    subs = char_diff(gt, corrected)
    if not subs:
        return None

    # Work on H=64 grayscale image (same as training)
    new_w = max(1, int(orig_w * 64 / orig_h))
    img64 = cv2.resize(img_bgr, (new_w, 64), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(img64, cv2.COLOR_BGR2GRAY)

    # Scale char ranges to H=64 image
    col_scale = new_w / orig_w
    scaled_ranges = [(c, int(cs * col_scale), int(ce * col_scale))
                     for c, cs, ce in char_ranges]

    # Estimate global skew from the H=64 image
    skew = estimate_skew(gray)

    patched = gray.copy()
    for pos, old_c, new_c in subs:
        if pos >= len(scaled_ranges):
            continue
        _, col_s, col_e = scaled_ranges[pos]
        region = gray[:, col_s:col_e] if col_e > col_s else None
        sw = estimate_stroke_width(region) if region is not None else 2.0
        region_w = max(col_e - col_s, 8)
        syn = render_synthetic_char(new_c, 64, region_w, font_paths,
                                    stroke_width=sw, skew_deg=skew)
        if syn is not None:
            patched = paste_char(patched, col_s, col_e, syn)

    return patched, corrected


# ---------------------------------------------------------------------------
# Write LMDB
# ---------------------------------------------------------------------------
def write_lmdb(output_path, samples):
    os.makedirs(output_path, exist_ok=True)
    map_size = max(1 << 31, len(samples) * 100 * 1024)
    env = lmdb.open(output_path, map_size=map_size)
    with env.begin(write=True) as txn:
        for i, (img, label) in enumerate(samples, 1):
            ok, buf = cv2.imencode('.png', img)
            if not ok:
                continue
            txn.put(f'image-{i:09d}'.encode(), buf.tobytes())
            txn.put(f'label-{i:09d}'.encode(), label.encode('utf-8'))
        txn.put(b'num-samples', str(len(samples)).encode())
    env.close()
    print(f'Written {len(samples)} samples → {output_path}')


# ---------------------------------------------------------------------------
# HTML viewer for QA
# ---------------------------------------------------------------------------
def write_html_viewer(records, out_path):
    import base64, html as html_mod
    rows = []
    for rec in records:
        def b64(arr):
            ok, buf = cv2.imencode('.png', arr)
            return base64.b64encode(buf.tobytes()).decode() if ok else ''

        orig_b64    = b64(rec['orig'])
        patched_b64 = b64(rec['patched'])
        gt   = html_mod.escape(rec['gt'])
        corr = html_mod.escape(rec['corrected'])
        subs = html_mod.escape(str(rec['subs']))

        rows.append(f'''
<div style="margin:12px 0;padding:10px;border:1px solid #ccc;border-radius:4px;font-family:monospace;font-size:12px">
  <div style="margin-bottom:4px"><b>Original:</b> {gt}</div>
  <img src="data:image/png;base64,{orig_b64}" style="height:64px;border:1px solid #aaa;display:block;margin-bottom:4px">
  <div style="margin-bottom:4px"><b>Patched ({subs}):</b> {corr}</div>
  <img src="data:image/png;base64,{patched_b64}" style="height:64px;border:1px solid #6a6;display:block">
</div>''')

    html_content = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Pixel-level Char Replacement ({len(records)} samples)</title>
<style>body{{margin:20px;background:#f8f8f8}}h1{{font-family:sans-serif}}</style>
</head><body>
<h1>Pixel-level Char Replacement — {len(records)} patched samples</h1>
{"".join(rows)}
</body></html>'''

    with open(out_path, 'w') as f:
        f.write(html_content)
    print(f'HTML viewer → {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',   default='checkpoints/exp1_full_iam/best.pt')
    parser.add_argument('--flagged',      default='results/doubao_train_flagged.csv')
    parser.add_argument('--img_dir',      default='data/iam_hf/train/images')
    parser.add_argument('--labels_csv',   default='data/iam_hf/train/labels.csv')
    parser.add_argument('--font_dir',     default='data/handwriting_fonts')
    parser.add_argument('--output_lmdb',  default='data/lmdb/train_patched')
    parser.add_argument('--output_html',  default='results/patch_viewer.html')
    parser.add_argument('--max_samples',  type=int, default=None,
                        help='Limit number of flagged samples to process (default: all)')
    parser.add_argument('--gpu',          default='6')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, device)

    # Load fonts
    font_paths = [os.path.join(args.font_dir, f)
                  for f in sorted(os.listdir(args.font_dir)) if f.endswith('.ttf')]
    print(f'Fonts: {len(font_paths)}')

    # Load flagged samples
    flagged = pd.read_csv(args.flagged)
    flagged = flagged[flagged['flagged'] == True].reset_index(drop=True)
    if args.max_samples:
        flagged = flagged.head(args.max_samples)
    print(f'Flagged samples to process: {len(flagged)}')

    # Load labels to get image paths
    labels = pd.read_csv(args.labels_csv)

    samples_out = []   # (patched_img, corrected_label)
    records_qa  = []   # for HTML viewer

    ok_count = fail_count = skip_count = 0

    for i, row in flagged.iterrows():
        idx  = int(row['idx'])          # already 0-based row index into labels.csv
        if idx < 0 or idx >= len(labels):
            skip_count += 1
            continue

        img_path  = labels.iloc[idx]['image_path']
        gt        = str(row['ground_truth'])
        corrected = str(row['corrected_text'])

        if gt == corrected:
            skip_count += 1
            continue

        subs = char_diff(gt, corrected)
        if not subs:
            skip_count += 1
            continue

        result = process_sample(img_path, gt, corrected, model, device, font_paths)
        if result is None:
            fail_count += 1
            continue

        patched, label_out = result
        samples_out.append((patched, label_out))
        ok_count += 1

        # Keep first 200 for HTML QA
        if len(records_qa) < 200:
            orig_img = cv2.imread(img_path)
            if orig_img is not None:
                orig_gray = cv2.cvtColor(
                    cv2.resize(orig_img, (patched.shape[1], 64)), cv2.COLOR_BGR2GRAY)
                records_qa.append({
                    'orig': orig_gray, 'patched': patched,
                    'gt': gt, 'corrected': label_out, 'subs': subs
                })

        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{len(flagged)}] ok={ok_count} fail={fail_count} skip={skip_count}')

    print(f'\nDone: ok={ok_count}  fail={fail_count}  skip={skip_count}')

    if samples_out:
        write_lmdb(args.output_lmdb, samples_out)
    if records_qa:
        write_html_viewer(records_qa, args.output_html)


if __name__ == '__main__':
    main()
