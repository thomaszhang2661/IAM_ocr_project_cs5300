"""
Generate an interactive HTML annotation page from doubao_check.py output.

Features:
  - Paginated display (PAGE_SIZE cards per page) — stays fast even for 2000+ samples
  - Per-sample: human_verdict dropdown + human_corrected text field
  - Annotations persist across page navigation (stored in JS)
  - "Export CSV" button → downloads audit CSV for build_clean_dataset.py
  - Filter by verdict type
  - Progress counter

Images saved as files (not base64) so HTML stays small.

Usage:
    python vlm_inference/make_review_html.py \
        --all_csv results/doubao_test_all.csv \
        --output  results/review_test.html \
        --split   test \
        --page_size 50
"""

import argparse
import json
import os

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

PAGE_SIZE_DEFAULT = 50


def save_images(df: pd.DataFrame, dataset, img_dir: str) -> None:
    os.makedirs(img_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Saving images'):
        idx   = int(row['idx'])
        fpath = os.path.join(img_dir, f'img_{idx:06d}.png')
        if not os.path.exists(fpath):
            dataset[idx]['image'].save(fpath)


def make_html(review_rows: list, all_rows: list, output_path: str,
              page_size: int = PAGE_SIZE_DEFAULT,
              title: str = 'IAM Annotation Review'):

    n_incorrect = sum(1 for r in review_rows if r.get('verdict') == 'INCORRECT')
    n_ambiguous = sum(1 for r in review_rows if r.get('verdict') == 'AMBIGUOUS')
    n_review    = n_incorrect + n_ambiguous

    all_data_json    = json.dumps(all_rows,    ensure_ascii=False)
    review_data_json = json.dumps(review_rows, ensure_ascii=False)

    style = """
* { box-sizing: border-box; }
body { font-family: Arial,sans-serif; margin:20px; background:#f0f2f5; color:#333; }
h1   { margin-bottom:4px; }
.subtitle { color:#888; font-size:0.9em; margin-bottom:14px; }
.stats { background:#fff; padding:12px 20px; border-radius:8px; margin-bottom:12px;
         display:flex; gap:24px; flex-wrap:wrap; box-shadow:0 1px 3px rgba(0,0,0,.08);
         align-items:center; }
.stat { text-align:center; }
.stat .num { font-size:1.6em; font-weight:bold; }
.stat .num.red   { color:#c62828; }
.stat .num.amber { color:#e65100; }
.stat .num.blue  { color:#1565c0; }
.stat .label { font-size:0.78em; color:#888; }
.toolbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:12px;
           background:#fff; padding:10px 16px; border-radius:8px;
           box-shadow:0 1px 3px rgba(0,0,0,.08); }
.toolbar label { cursor:pointer; font-size:0.92em; display:flex; align-items:center; gap:4px; }
.export-btn { background:#1565c0; color:#fff; border:none; padding:7px 18px;
              border-radius:6px; cursor:pointer; font-size:0.92em; }
.export-btn:hover { background:#0d47a1; }
.pagination { display:flex; gap:6px; align-items:center; margin-bottom:14px;
              background:#fff; padding:8px 16px; border-radius:8px;
              box-shadow:0 1px 3px rgba(0,0,0,.08); flex-wrap:wrap; }
.pagination button { padding:4px 12px; border:1px solid #ccc; border-radius:4px;
                     background:#fff; cursor:pointer; font-size:0.88em; }
.pagination button:hover:not(:disabled) { background:#e3f2fd; border-color:#1565c0; }
.pagination button:disabled { color:#bbb; cursor:default; }
.pagination .page-info { font-size:0.88em; color:#666; margin:0 6px; }
.pagination select { padding:4px 8px; border:1px solid #ccc; border-radius:4px;
                     font-size:0.88em; }
.card { background:#fff; border-radius:8px; margin-bottom:12px;
        padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,.1); }
.card.INCORRECT { border-left:5px solid #c62828; }
.card.AMBIGUOUS { border-left:5px solid #e65100; }
.card img { max-width:100%; height:auto; display:block; margin:8px 0;
            border:1px solid #e0e0e0; background:#fafafa; border-radius:4px; }
.row { display:flex; gap:8px; align-items:baseline; margin:4px 0; font-size:0.9em; }
.key { font-weight:bold; min-width:90px; color:#555; flex-shrink:0; }
.val { font-family:monospace; word-break:break-word; }
.val.gt   { background:#e8eaf6; padding:2px 6px; border-radius:3px; }
.val.err  { background:#ffebee; color:#c62828; padding:2px 6px; border-radius:3px; }
.val.corr { background:#e8f5e9; color:#1b5e20; padding:2px 6px; border-radius:3px; }
.val.amb  { background:#fff8e1; color:#e65100; padding:2px 6px; border-radius:3px; }
.idx { color:#aaa; font-size:0.76em; margin-bottom:6px; }
.annotation-area { margin-top:10px; padding-top:10px; border-top:1px solid #f0f0f0;
                   display:flex; gap:10px; align-items:flex-start; flex-wrap:wrap; }
.verdict-select { padding:5px 10px; font-size:0.88em; border:1px solid #ccc;
                  border-radius:5px; background:#fff; cursor:pointer; min-width:170px; }
.verdict-select.confirmed_correct { border-color:#2e7d32; background:#e8f5e9; color:#1b5e20; }
.verdict-select.confirmed_error   { border-color:#c62828; background:#ffebee; color:#c62828; }
.verdict-select.skip              { border-color:#aaa;    background:#f5f5f5; color:#888; }
.correction-input { flex:1; min-width:200px; padding:5px 8px; font-size:0.88em;
                    font-family:monospace; border:1px solid #ccc; border-radius:5px; }
.correction-input:focus { outline:none; border-color:#1565c0; box-shadow:0 0 0 2px #bbdefb; }
"""

    script = f"""
const allData    = {all_data_json};
const reviewData = {review_data_json};
const IMG_DIR    = 'review_images/';
let   PAGE_SIZE  = {page_size};
let   currentPage = 1;
let   showInc = true, showAmb = false;

// annotation state: idx -> {{human_verdict, human_corrected}}
const annotations = {{}};
reviewData.forEach(r => {{
    annotations[r.idx] = {{ human_verdict: '', human_corrected: '' }};
}});

function getFiltered() {{
    return reviewData.filter(r =>
        (r.verdict === 'INCORRECT' && showInc) ||
        (r.verdict === 'AMBIGUOUS' && showAmb)
    );
}}

function totalPages() {{
    return Math.max(1, Math.ceil(getFiltered().length / PAGE_SIZE));
}}

function renderPage(page) {{
    currentPage = Math.max(1, Math.min(page, totalPages()));
    const filtered = getFiltered();
    const start = (currentPage - 1) * PAGE_SIZE;
    const pageItems = filtered.slice(start, start + PAGE_SIZE);

    const container = document.getElementById('cards');
    container.innerHTML = '';
    pageItems.forEach(r => {{
        container.appendChild(buildCard(r));
    }});

    updatePagination();
    updateProgress();
    window.scrollTo(0, 0);
}}

function buildCard(r) {{
    const ann = annotations[r.idx];
    const isInc = r.verdict === 'INCORRECT';
    const color = isInc ? '#c62828' : '#e65100';

    const div = document.createElement('div');
    div.className = `card ${{r.verdict}}`;
    div.dataset.idx = r.idx;

    let detailHtml = '';
    if (isInc) {{
        detailHtml += `<div class="row"><span class="key">Errors:</span><span class="val err">${{esc(r.reason)}}</span></div>`;
        if (r.corrected_text)
            detailHtml += `<div class="row"><span class="key">Corrected:</span><span class="val corr">${{esc(r.corrected_text)}}</span></div>`;
    }} else {{
        detailHtml += `<div class="row"><span class="key">Reason:</span><span class="val amb">${{esc(r.reason)}}</span></div>`;
    }}

    const selVal = ann.human_verdict || (isInc ? 'confirmed_error' : '');
    if (!ann.human_verdict && isInc) ann.human_verdict = 'confirmed_error';

    const corrVal = ann.human_corrected !== undefined ? ann.human_corrected
                    : (isInc ? r.corrected_text : '');

    div.innerHTML = `
      <div class="idx">#${{r.idx}} &nbsp;·&nbsp; ${{r.split}} &nbsp;·&nbsp;
        <span style="color:${{color}}">${{r.verdict}}</span></div>
      <img src="${{IMG_DIR}}img_${{String(r.idx).padStart(6,'0')}}.png"
           alt="sample ${{r.idx}}" loading="lazy">
      <div class="row"><span class="key">GT:</span><span class="val gt">${{esc(r.ground_truth)}}</span></div>
      ${{detailHtml}}
      <div class="annotation-area">
        <select class="verdict-select ${{selVal}}"
                onchange="onVerdictChange(${{r.idx}}, this)">
          <option value="">— verdict —</option>
          <option value="confirmed_correct" ${{selVal==='confirmed_correct'?'selected':''}}>confirmed_correct</option>
          <option value="confirmed_error"   ${{selVal==='confirmed_error'  ?'selected':''}}>confirmed_error</option>
          <option value="skip"              ${{selVal==='skip'             ?'selected':''}}>skip</option>
        </select>
        <input class="correction-input" type="text"
               placeholder="human_corrected (blank = use model suggestion)"
               value="${{esc(corrVal)}}"
               oninput="onCorrectionChange(${{r.idx}}, this)">
      </div>`;
    return div;
}}

function esc(s) {{
    return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

function onVerdictChange(idx, sel) {{
    annotations[idx].human_verdict = sel.value;
    sel.className = 'verdict-select ' + sel.value;
    updateProgress();
}}

function onCorrectionChange(idx, inp) {{
    annotations[idx].human_corrected = inp.value;
}}

function countReviewed() {{
    return Object.values(annotations).filter(a => a.human_verdict !== '').length;
}}

function updateProgress() {{
    const n = countReviewed();
    const total = reviewData.length;
    document.getElementById('reviewed_count').textContent = n;
    document.getElementById('progress_bar').style.width = Math.round(n/total*100) + '%';
}}

function updatePagination() {{
    const tp = totalPages();
    const filtered = getFiltered();
    document.getElementById('page_info').textContent =
        `Page ${{currentPage}} / ${{tp}}  (${{filtered.length}} items)`;
    document.getElementById('btn_first').disabled = currentPage <= 1;
    document.getElementById('btn_prev').disabled  = currentPage <= 1;
    document.getElementById('btn_next').disabled  = currentPage >= tp;
    document.getElementById('btn_last').disabled  = currentPage >= tp;
    document.getElementById('page_jump').value = currentPage;
    document.getElementById('page_jump').max   = tp;
}}

function jumpPage() {{
    const v = parseInt(document.getElementById('page_jump').value);
    if (!isNaN(v)) renderPage(v);
}}

function changePageSize() {{
    PAGE_SIZE = parseInt(document.getElementById('page_size_sel').value);
    renderPage(1);
}}

function applyFilter() {{
    showInc = document.getElementById('chk_incorrect').checked;
    showAmb = document.getElementById('chk_ambiguous').checked;
    renderPage(1);
}}

function exportCSV() {{
    const header = ['idx','split','ground_truth','doubao_verdict',
                    'doubao_reason','doubao_corrected',
                    'human_verdict','human_corrected'];
    const escCSV = v => {{
        const s = String(v ?? '');
        return (s.includes(',') || s.includes('"') || s.includes('\\n'))
            ? '"' + s.replace(/"/g, '""') + '"' : s;
    }};
    const lines = [header.join(',')];
    allData.forEach(r => {{
        const ann = annotations[r.idx] || {{human_verdict:'',human_corrected:''}};
        lines.push([
            r.idx, escCSV(r.split), escCSV(r.ground_truth),
            escCSV(r.verdict), escCSV(r.reason), escCSV(r.corrected_text),
            escCSV(ann.human_verdict), escCSV(ann.human_corrected),
        ].join(','));
    }});
    const blob = new Blob([lines.join('\\n')], {{type:'text/csv'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `audit_${{allData[0]?.split||'export'}}.csv`;
    a.click();
}}

// Init
window.onload = () => renderPage(1);
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>{style}</style>
</head>
<body>
  <h1>{title}</h1>
  <p class="subtitle">Review INCORRECT/AMBIGUOUS samples · Set verdict · Export CSV when done.</p>

  <div class="stats">
    <div class="stat"><div class="num red">{n_incorrect}</div><div class="label">INCORRECT</div></div>
    <div class="stat"><div class="num amber">{n_ambiguous}</div><div class="label">AMBIGUOUS</div></div>
    <div class="stat"><div class="num blue" id="reviewed_count">0</div><div class="label">Reviewed</div></div>
    <div class="stat" style="flex:1;min-width:200px;">
      <div style="background:#e0e0e0;border-radius:4px;height:8px;overflow:hidden;">
        <div id="progress_bar" style="background:#1565c0;height:100%;width:0%;transition:width .3s;"></div>
      </div>
      <div class="label" style="margin-top:4px;">Progress ({n_review} to review)</div>
    </div>
  </div>

  <div class="toolbar">
    <b>Show:</b>
    <label><input type="checkbox" id="chk_incorrect" checked onchange="applyFilter()"> INCORRECT ({n_incorrect})</label>
    <label><input type="checkbox" id="chk_ambiguous" onchange="applyFilter()"> AMBIGUOUS ({n_ambiguous})</label>
    <span style="margin-left:8px;font-size:0.88em;color:#888;">Per page:
      <select id="page_size_sel" onchange="changePageSize()">
        <option value="25">25</option>
        <option value="50" selected>50</option>
        <option value="100">100</option>
      </select>
    </span>
    <button class="export-btn" onclick="exportCSV()">Export CSV</button>
  </div>

  <div class="pagination">
    <button id="btn_first" onclick="renderPage(1)">&#171; First</button>
    <button id="btn_prev"  onclick="renderPage(currentPage-1)">&#8249; Prev</button>
    <span class="page-info" id="page_info"></span>
    <input id="page_jump" type="number" min="1" style="width:55px;padding:4px;border:1px solid #ccc;border-radius:4px;font-size:0.88em;"
           onchange="jumpPage()">
    <button id="btn_next" onclick="renderPage(currentPage+1)">Next &#8250;</button>
    <button id="btn_last" onclick="renderPage(totalPages())">Last &#187;</button>
  </div>

  <div id="cards"></div>

  <div class="pagination" style="margin-top:4px;">
    <button id="btn_first2" onclick="renderPage(1);document.getElementById('btn_first').scrollIntoView()">&#171; First</button>
    <button onclick="renderPage(currentPage-1);window.scrollTo(0,0)">&#8249; Prev</button>
    <span class="page-info" id="page_info2"></span>
    <button onclick="renderPage(currentPage+1);window.scrollTo(0,0)">Next &#8250;</button>
    <button onclick="renderPage(totalPages());window.scrollTo(0,0)">Last &#187;</button>
  </div>

  <script>
    // sync bottom pagination info
    const origUpdatePag = updatePagination;
    function updatePagination() {{
      origUpdatePag && origUpdatePag();
      const el = document.getElementById('page_info2');
      if (el) el.textContent = document.getElementById('page_info').textContent;
    }}
  </script>
  <script>{script}</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = os.path.getsize(output_path) / 1024
    print(f'HTML: {output_path}  ({size_kb:.0f} KB)')
    print(f'Open: file://{os.path.abspath(output_path)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_csv',   required=True)
    parser.add_argument('--output',    default='results/review.html')
    parser.add_argument('--split',     default='test')
    parser.add_argument('--page_size', type=int, default=PAGE_SIZE_DEFAULT)
    args = parser.parse_args()

    df = pd.read_csv(args.all_csv)
    df_review = df[df['verdict'].isin(['INCORRECT', 'AMBIGUOUS'])].copy()
    print(f'Review samples: {len(df_review)} '
          f'(INCORRECT={(df["verdict"]=="INCORRECT").sum()}, '
          f'AMBIGUOUS={(df["verdict"]=="AMBIGUOUS").sum()})')

    hf_split = 'validation' if args.split == 'val' else args.split
    print(f'Loading images ({hf_split})...')
    dataset = load_dataset('Teklia/IAM-line', split=hf_split)

    img_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)),
                           f'review_images_{args.split}')
    save_images(df_review, dataset, img_dir)

    # review rows (cards)
    review_rows = []
    for _, row in df_review.iterrows():
        idx = int(row['idx'])
        review_rows.append({
            'idx':            idx,
            'split':          row.get('split', args.split),
            'ground_truth':   str(row.get('ground_truth', '') or ''),
            'verdict':        str(row.get('verdict', '') or ''),
            'reason':         str(row.get('reason', '') or ''),
            'corrected_text': str(row.get('corrected_text', '') or ''),
        })

    # all rows for CSV export
    all_rows = []
    for _, row in df.iterrows():
        all_rows.append({
            'idx':            int(row['idx']),
            'split':          str(row.get('split', args.split) or ''),
            'ground_truth':   str(row.get('ground_truth', '') or ''),
            'verdict':        str(row.get('verdict', '') or ''),
            'reason':         str(row.get('reason', '') or ''),
            'corrected_text': str(row.get('corrected_text', '') or ''),
        })

    make_html(review_rows, all_rows, args.output,
              page_size=args.page_size,
              title=f'IAM Annotation Review — {args.split}')


if __name__ == '__main__':
    main()
