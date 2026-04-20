"""
Analyze VLM annotation results to find character-level confusion pairs.

Outputs:
  results/vlm_char_confusion.csv   — confusion pair counts across all splits
  results/vlm_confusion_report.html — visual HTML with images for top confusions
"""

import base64
import difflib
import io
import os
from collections import Counter, defaultdict

import cv2
import lmdb
import numpy as np
import pandas as pd
from PIL import Image


# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
SPLITS = {
    'train': ('./results/doubao_train_flagged.csv', './data/lmdb/train'),
    'val':   ('./results/doubao_val_flagged.csv',   './data/lmdb/val'),
    'test':  ('./results/doubao_test_flagged.csv',  './data/lmdb/test'),
}
OUT_CSV  = './results/vlm_char_confusion.csv'
OUT_HTML = './results/vlm_confusion_report.html'
MAX_EXAMPLES_PER_PAIR = 5  # images per confusion pair in HTML
TOP_N_PAIRS = 40           # how many confusion pairs to show


# -----------------------------------------------------------------------
# Character-level alignment
# -----------------------------------------------------------------------
def char_substitutions(a, b):
    """
    Return list of (wrong_char, correct_char) substitutions found by
    aligning string a (ground_truth) to string b (corrected_text).
    Only substitutions, not insertions/deletions.
    """
    subs = []
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Align the replaced segment char by char (zip truncates to shorter)
            for ca, cb in zip(a[i1:i2], b[j1:j2]):
                if ca != cb:
                    subs.append((ca, cb))
    return subs


# -----------------------------------------------------------------------
# Load image from LMDB
# -----------------------------------------------------------------------
def load_image_from_lmdb(env, idx):
    """Load image at 1-based idx from LMDB. Returns numpy H×W uint8 or None."""
    with env.begin() as txn:
        buf = txn.get(f'image-{idx:09d}'.encode())
    if buf is None:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


def img_to_base64(img, max_w=600):
    """Convert grayscale numpy array to base64 PNG for HTML embedding."""
    h, w = img.shape
    if w > max_w:
        new_h = int(h * max_w / w)
        img = cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# -----------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------
def main():
    os.makedirs('./results', exist_ok=True)

    # pair → list of (split, idx, gt, corrected)
    pair_examples = defaultdict(list)
    pair_counts   = Counter()

    print('Analyzing character substitutions...')
    for split, (csv_path, lmdb_path) in SPLITS.items():
        df = pd.read_csv(csv_path)
        print(f'  {split}: {len(df)} flagged samples')

        for _, row in df.iterrows():
            gt  = str(row['ground_truth'])   if pd.notna(row['ground_truth'])   else ''
            cor = str(row['corrected_text'])  if pd.notna(row['corrected_text']) else ''
            if not gt or not cor:
                continue

            subs = char_substitutions(gt, cor)
            for wrong, correct in subs:
                pair = (wrong, correct)
                pair_counts[pair] += 1
                pair_examples[pair].append({
                    'split': split,
                    'idx':   int(row['idx']),
                    'gt':    gt,
                    'cor':   cor,
                    'lmdb':  lmdb_path,
                })

    # -----------------------------------------------------------------------
    # CSV output
    # -----------------------------------------------------------------------
    rows = []
    for (wrong, correct), count in pair_counts.most_common():
        rows.append({
            'wrong_char':   wrong,
            'correct_char': correct,
            'count':        count,
            'wrong_repr':   repr(wrong),
            'correct_repr': repr(correct),
            'note': _describe_confusion(wrong, correct),
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f'\nSaved confusion CSV → {OUT_CSV}')
    print(f'Total unique confusion pairs: {len(rows)}')
    print('\nTop 20 pairs:')
    print(df_out.head(20).to_string(index=False))

    # -----------------------------------------------------------------------
    # HTML output with images
    # -----------------------------------------------------------------------
    print(f'\nGenerating HTML with images → {OUT_HTML}')

    # Open all LMDBs
    lmdb_envs = {}
    for split, (_, lmdb_path) in SPLITS.items():
        lmdb_envs[split] = lmdb.open(lmdb_path, readonly=True, lock=False)

    top_pairs = [row for row in rows[:TOP_N_PAIRS]]

    html_rows = []
    for rank, row in enumerate(top_pairs, 1):
        wrong   = row['wrong_char']
        correct = row['correct_char']
        count   = row['count']
        pair    = (wrong, correct)
        note    = row['note']

        examples = pair_examples[pair][:MAX_EXAMPLES_PER_PAIR]

        imgs_html = ''
        for ex in examples:
            env = lmdb_envs[ex['split']]
            img = load_image_from_lmdb(env, ex['idx'])
            if img is None:
                continue
            b64  = img_to_base64(img)
            # Highlight the difference in text
            gt_hl  = _highlight_char(ex['gt'],  wrong)
            cor_hl = _highlight_char(ex['cor'],  correct)
            imgs_html += f'''
            <div class="example">
              <img src="data:image/png;base64,{b64}" alt="sample">
              <div class="labels">
                <span class="gt">GT: {gt_hl}</span>
                <span class="cor">→ {cor_hl}</span>
                <span class="sp">[{ex["split"]} #{ex["idx"]}]</span>
              </div>
            </div>'''

        pair_id = f'pair_{rank}'
        html_rows.append(f'''
    <tr>
      <td class="rank">{rank}</td>
      <td class="char wrong">{_safe_html(wrong)}</td>
      <td class="arrow">→</td>
      <td class="char correct">{_safe_html(correct)}</td>
      <td class="count">{count}</td>
      <td class="note">{note}</td>
      <td>
        <button onclick="toggle('{pair_id}')">show / hide</button>
        <div id="{pair_id}" class="examples" style="display:none">{imgs_html}</div>
      </td>
    </tr>''')

    html = _html_template('\n'.join(html_rows))
    with open(OUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Done → {OUT_HTML}')

    for env in lmdb_envs.values():
        env.close()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _describe_confusion(wrong, correct):
    """Short human-readable description of a confusion pair."""
    categories = {
        'digit':  set('0123456789'),
        'upper':  set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
        'lower':  set('abcdefghijklmnopqrstuvwxyz'),
        'punct':  set(' !"#&\'()*+,-./0123456789:;?'),
    }
    def cat(c):
        if c.isdigit(): return 'digit'
        if c.isupper(): return 'upper'
        if c.islower(): return 'lower'
        if c == ' ':    return 'space'
        return 'punct'
    cw, cc = cat(wrong), cat(correct)
    if cw == cc:
        return f'{cw}–{cc} swap'
    return f'{cw} mistaken as {cc}'


def _safe_html(c):
    return (c.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
             .replace('"','&quot;') or '(space)' if c == ' ' else
            c.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'))


def _highlight_char(text, char):
    """Wrap all occurrences of char in <mark>."""
    safe = text.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    safe_c = char.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    return safe.replace(safe_c, f'<mark>{safe_c}</mark>')


def _html_template(table_rows):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VLM Character Confusion Analysis</title>
<style>
  body {{ font-family: monospace; margin: 20px; background: #fafafa; }}
  h1 {{ font-size: 1.4em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; vertical-align: top; }}
  th {{ background: #2c3e50; color: white; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  .rank {{ text-align: center; color: #888; width: 40px; }}
  .char {{ font-size: 1.6em; font-weight: bold; text-align: center; width: 50px; }}
  .wrong {{ color: #c0392b; }}
  .correct {{ color: #27ae60; }}
  .arrow {{ text-align: center; font-size: 1.4em; color: #888; width: 30px; }}
  .count {{ text-align: center; font-weight: bold; width: 60px; }}
  .note {{ color: #555; font-size: 0.85em; width: 180px; }}
  .examples {{ margin-top: 8px; }}
  .example {{ border: 1px solid #ccc; border-radius: 4px; padding: 6px;
              margin: 4px 0; background: white; }}
  .example img {{ max-width: 600px; display: block; border: 1px solid #eee; }}
  .labels {{ font-size: 0.82em; margin-top: 4px; }}
  .gt  {{ color: #c0392b; margin-right: 12px; }}
  .cor {{ color: #27ae60; margin-right: 12px; }}
  .sp  {{ color: #999; }}
  mark {{ background: #ffe066; padding: 0 1px; }}
  button {{ font-size: 0.8em; cursor: pointer; padding: 2px 8px; }}
</style>
</head>
<body>
<h1>VLM Character Confusion Analysis — Top {TOP_N_PAIRS} Pairs</h1>
<p>Red = character in IAM ground truth &nbsp;|&nbsp; Green = VLM correction &nbsp;|&nbsp;
   Count = occurrences across train+val+test</p>
<table>
  <thead>
    <tr>
      <th>#</th><th>Wrong</th><th></th><th>Correct</th>
      <th>Count</th><th>Category</th><th>Examples</th>
    </tr>
  </thead>
  <tbody>
{table_rows}
  </tbody>
</table>
<script>
function toggle(id) {{
  var el = document.getElementById(id);
  el.style.display = (el.style.display === 'none') ? 'block' : 'none';
}}
</script>
</body>
</html>'''


if __name__ == '__main__':
    main()
