"""
Synthetic handwriting line generation for IAM CRNN training.

Text sources (3 layers):
  1. IAM train labels       — real IAM-style sentences (best domain match)
  2. Wikipedia sentences    — broad English vocabulary
  3. Targeted fill-up       — template sentences dense in rare chars
                              (digits, uppercase, punctuation)
                              generated until each rare char hits MIN_COUNT

After generation, saves:
  - data/lmdb/train_synth/          LMDB (same format as train/)
  - results/synth_char_freq.csv     per-character count in synthetic set
  - results/synth_char_freq.png     bar chart (log scale)

Usage:
    python data/generate_synthetic.py
    python data/generate_synthetic.py --n_base 15000 --output_lmdb data/lmdb/train_synth
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import cv2
import lmdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IAM_ALPHABET = (
    ' !"#&\'()*+,-./0123456789:;?'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abcdefghijklmnopqrstuvwxyz'
)
VALID_CHARS = set(IAM_ALPHABET)

# Characters that are rare in IAM (<500 occurrences) — we want to boost these
RARE_CHARS = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?#&*+/;:()')
MIN_RARE_COUNT = 500   # target per rare char in synthetic set

FONT_DIR = os.path.join(os.path.dirname(__file__), 'handwriting_fonts')
IMG_H    = 64


# ---------------------------------------------------------------------------
# Text source 1: IAM train labels
# ---------------------------------------------------------------------------
def load_iam_texts():
    import pandas as pd
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'iam_hf/train/labels.csv'))
    return df['text'].dropna().tolist()


# ---------------------------------------------------------------------------
# Text source 2: Wikipedia sentences
# ---------------------------------------------------------------------------
def fetch_wikipedia_sentences(n_articles=30, seed=42):
    """
    Fetch random Wikipedia article summaries and split into sentences.
    Uses a short timeout per request; skips on failure.
    """
    try:
        import wikipedia
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError

        wikipedia.set_lang('en')
        random.seed(seed)

        sentences = []
        tried = set()
        max_tries = n_articles * 3

        for _ in range(max_tries):
            if len(tried) >= n_articles:
                break
            try:
                # Hard 5-second timeout per article
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(5)
                title = wikipedia.random(1)
                if title in tried:
                    signal.alarm(0)
                    continue
                tried.add(title)
                summary = wikipedia.summary(title, sentences=6, auto_suggest=False)
                signal.alarm(0)
                for sent in re.split(r'(?<=[.!?])\s+', summary):
                    sent = ''.join(c for c in sent.strip() if c in VALID_CHARS)
                    if 20 <= len(sent) <= 120:
                        sentences.append(sent)
            except Exception:
                signal.alarm(0)
                continue

        print(f'  Wikipedia: {len(sentences)} sentences from {len(tried)} articles')
        return sentences
    except Exception as e:
        print(f'  Wikipedia unavailable ({e}), skipping')
        return []


# ---------------------------------------------------------------------------
# Text source 3: Targeted rare-character sentences
# ---------------------------------------------------------------------------
# Pre-written sentence pools dense in each rare category

_DIGIT_SENTENCES = [
    "In {y}, about {n1} people attended the {adj} conference.",
    "The score was {n1}-{n2} after {n3} minutes of play in {y}.",
    "Call {phone} between {h1}:00 and {h2}:30 on weekdays.",
    "Chapter {n1}, Section {n2}.{n3}: see pages {n4} to {n5}.",
    "Order #{n1}: {n2} units at ${n3}.{n4} each, total ${n5}.",
    "The temperature reached {n1}.{n2} degrees on {day} {n3}, {y}.",
    "Train departs at {h1}:{m1} and arrives at {h2}:{m2}.",
    "Population of {city} grew from {n1},{n2}00 to {n3},{n4}00 between {y1} and {y2}.",
    "Reference: Vol. {n1}, No. {n2}, pp. {n3}-{n4}, {y}.",
    "The flight {n1}{L1} departs Gate {n2} at {h1}:{m1}.",
]

_UPPER_SENTENCES = [
    "DR. {F} {L} OF {C} UNIVERSITY PRESENTED THE {adj} REPORT.",
    "MR. {F} {L} AND MRS. {F2} {L2} SIGNED THE CONTRACT.",
    "{F} {L} JOINED {C} CORP. AS CEO IN {y}.",
    "THE {C} COMMITTEE MET ON {day} TO DISCUSS THE {adj} PLAN.",
    "NOTICE: ALL {C} STAFF MUST REGISTER BY {day} {n}.",
    "FROM: {F} {L}, {C} DIVISION. RE: {adj} POLICY UPDATE.",
    "ATTN: {F} {L} — PLEASE REVIEW THE ATTACHED {adj} DOCUMENT.",
    "CHAPTER {R}: THE {adj} OF {C}.",
    # Z/Q/K explicit coverage
    "ZONE {n1} & ZONE {n2}: QUICK QUARTERLY REVIEW BY {day}.",
    "Q{n1}: QUALITY CHECK FOR ZONE {L1}? KENNETH {L} TO CONFIRM.",
    "ZACHARY {L} & QUINCY {L2} QUALIFIED FOR ZONE {n1} QUOTA.",
    "KEY TASK #{n1}: QUIZ RESULTS FOR ZONE {L1} — QUICK REVIEW?",
    "Q: DOES ZONE {n1} QUALIFY? A: YES, SUBJECT TO REVIEW.",
    "ZAKK {L} REQUESTED {n1} UNITS; QUOTA EXCEEDED BY {n2}.",
]

_PUNCT_SENTENCES = [
    'He said, "I don\'t know," and walked away.',
    "Wait -- are you sure? Yes! I\'m absolutely certain.",
    "The results (see Table {n}) show a significant difference.",
    "Items needed: flour, sugar, eggs & butter.",
    "Q: What is {n1} + {n2}? A: {n3}.",
    "Dear Sir/Madam, please find enclosed the {adj} report.",
    "Note: prices include tax (see Section {n}.{n2}).",
    'She replied: "That\'s impossible!" and hung up.',
    "The contract -- signed on {day} -- expires in {y}.",
    "Ref. #: {n1}/{n2}-{y}. Status: PENDING.",
    "Hours: Mon-Fri {h1}:{m1}-{h2}:{m2}; Sat {h1}:{m1}-{h3}:00.",
    # & coverage
    "Bread & butter, salt & pepper, oil & vinegar.",
    "{F} & {L} Ltd. was founded in {y} by {F2} & {L2}.",
    "Terms & conditions apply; see Section {n} for details.",
    "The {adj} agreement between {C} & {C} was signed on {day}.",
    # ; coverage
    "First, gather the data; second, clean it; third, analyse it.",
    "The report was late; however, the results were accurate.",
    "He arrived early; she arrived late; they met at noon.",
    # ? coverage
    "What time does the train depart? Is there a direct service?",
    "Can you confirm the date? Who signed the contract?",
    "Why was the report delayed? When will it be ready?",
    # + coverage
    "Total: {n1} + {n2} + {n3} = {n4}.",
    "The formula is: result = x + y + z * {n1}.",
    "Section {n1} + Appendix {n2} covers all cases.",
    # Z, Q, K coverage
    "ZONE {n1}: QUICK RESPONSE REQUIRED BY {F} {L}.",
    "Q{n1}: Quarterly results for {C} ZONE exceeded targets.",
    "TASK {n1}: KENNETH {L} and ZACHARY {L2} to review by {day}.",
    "QUIZ #{n1}: Questions {n2}-{n3} relate to ZONE {L1}.",
    "KEY QUESTION: Does {C} qualify under REGULATION {n1}/{y}?",
    "KAZAKH ZONE {n1} — QUOTA: {n2} units quarterly.",
    # * coverage
    "Note: items marked * require approval; ** require sign-off.",
    "* See footnote {n1}. ** Refer to Appendix {n2}.",
    "Result: {n1} * {n2} = {n3} (starred items only).",
    # / coverage
    "Date: {n1}/{n2}/{y}. File: {n3}/{adj}/report.pdf.",
    "Ratio: {n1}/{n2} per unit, or {n3}/{n4} per batch.",
    "Tel: {h1}/{n1}-{n2}. Alt: {h2}/{n3}-{n4}.",
]

def _fill(tmpl):
    names_f = ['James','John','Robert','Mary','Patricia','William','David',
               'Charles','Thomas','George','Edward','Margaret','Elizabeth']
    names_l = ['Smith','Johnson','Williams','Brown','Jones','Davis','Miller',
               'Wilson','Taylor','Anderson','Thomas','Jackson','White']
    cities  = ['London','Oxford','Cambridge','Bristol','Edinburgh','Cardiff']
    adjs    = ['annual','quarterly','final','preliminary','revised','official',
               'general','special','regional','national']
    days    = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    roman   = list('IVXLCDM')

    try:
        s = tmpl.format(
            y   =random.randint(1950, 2024),
            y1  =random.randint(1950, 2000),
            y2  =random.randint(2001, 2024),
            n   =random.randint(1, 99),
            n1  =random.randint(1, 999),
            n2  =random.randint(1, 99),
            n3  =random.randint(1, 999),
            n4  =random.randint(1, 99),
            n5  =random.randint(100, 9999),
            h1  =random.randint(8, 12),
            h2  =random.randint(13, 18),
            h3  =random.randint(19, 22),
            m1  =random.choice(['00','15','30','45']),
            m2  =random.choice(['00','15','30','45']),
            phone=f"{random.randint(100,999)}-{random.randint(1000,9999)}",
            F   =random.choice(names_f),
            F2  =random.choice(names_f),
            L   =random.choice(names_l),
            L2  =random.choice(names_l),
            C   =random.choice(cities).upper(),
            adj =random.choice(adjs),
            day =random.choice(days),
            city=random.choice(cities),
            R   =random.choice(roman),
            L1  =random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
        )
        return ''.join(c for c in s if c in VALID_CHARS)
    except Exception:
        return ''


def generate_targeted(needed_counts):
    """
    Generate sentences until every rare char in needed_counts is satisfied.
    needed_counts: {char: remaining_count_needed}
    Returns list of sentences.
    """
    sentences = []
    counter = Counter()
    max_iter = 100_000

    for _ in range(max_iter):
        # Pick template pool based on which chars still need filling
        still_needed = {c: n for c, n in needed_counts.items()
                        if counter[c] < n}
        if not still_needed:
            break

        # Build weighted pool: include all template pools that cover still-needed chars
        pool = []
        if any(c.isdigit() for c in still_needed):
            pool += _DIGIT_SENTENCES * 2
        if any(c.isupper() for c in still_needed):
            pool += _UPPER_SENTENCES * 2
        if any(c in still_needed for c in '#&*+/;?'):
            pool += _PUNCT_SENTENCES * 3
        if not pool:
            pool = _DIGIT_SENTENCES + _UPPER_SENTENCES + _PUNCT_SENTENCES

        tmpl = random.choice(pool)
        s = _fill(tmpl)
        if len(s) < 10:
            continue

        sentences.append(s)
        counter.update(s)

    print(f'  Targeted: generated {len(sentences)} sentences')
    # Report coverage
    for c in sorted(needed_counts):
        got = counter[c]
        need = needed_counts[c]
        status = 'OK' if got >= need else f'SHORT by {need-got}'
        print(f'    {repr(c):4s}: {got:5d} / {need}  {status}')
    return sentences


# ---------------------------------------------------------------------------
# Build full text pool
# ---------------------------------------------------------------------------
def load_rare_texts(path):
    """Load Doubao-generated rare-char sentences if available."""
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    # Filter to valid IAM chars only
    filtered = []
    for line in lines:
        cleaned = ''.join(c for c in line if c in VALID_CHARS)
        if len(cleaned) >= 10:
            filtered.append(cleaned)
    print(f'  Rare texts (Doubao): {len(filtered)} sentences from {path}')
    return filtered


def build_text_pool(rare_texts_path='./data/rare_texts.txt'):
    print('Loading IAM texts...')
    iam = load_iam_texts()
    print(f'  IAM texts: {len(iam)}')

    # Wikipedia fetch disabled (network not available on this server)
    wiki = []
    print('  Wikipedia: skipped (network unavailable)')

    # Load Doubao-generated rare texts if available
    rare_api = load_rare_texts(rare_texts_path)

    # Check what's still rare after IAM + wiki + rare_api
    all_text = ''.join(iam + wiki + rare_api)
    counts = Counter(all_text)
    needed = {c: max(0, MIN_RARE_COUNT - counts.get(c, 0))
              for c in RARE_CHARS if counts.get(c, 0) < MIN_RARE_COUNT}
    print(f'\nRare chars still below {MIN_RARE_COUNT} after IAM+Wiki+RareAPI: '
          f'{len(needed)} chars')
    for c, n in sorted(needed.items()):
        print(f'  {repr(c):4s}: have {counts.get(c,0):4d}, need {n} more')

    if needed:
        print('\nGenerating template-based targeted sentences for remaining gaps...')
        targeted = generate_targeted(needed)
    else:
        targeted = []
        print('\nAll rare chars satisfied — no template generation needed.')

    # Combine: IAM × 2 (domain match), rare_api × 3, targeted × 2
    pool = iam * 2 + rare_api * 3 + targeted * 2
    random.shuffle(pool)
    print(f'\nTotal text pool: {len(pool)} sentences')
    return pool


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------
def get_fonts():
    if not os.path.exists(FONT_DIR):
        raise FileNotFoundError(f'Font dir not found: {FONT_DIR}')
    return [os.path.join(FONT_DIR, f)
            for f in sorted(os.listdir(FONT_DIR)) if f.endswith('.ttf')]


def build_font_coverage(font_paths):
    """
    Returns:
      char_to_fonts: {char: [font_paths that have this char]}
      font_sizes:    {font_path: optimal PIL font size for TARGET_H}
    """
    from PIL import ImageFont
    from fontTools.ttLib import TTFont

    PADDING = 4
    char_to_fonts = {c: [] for c in IAM_ALPHABET}

    font_sizes = {}
    for fp in font_paths:
        # Find font size
        lo, hi = 8, IMG_H * 2
        for _ in range(20):
            mid = (lo + hi) // 2
            f = ImageFont.truetype(fp, size=mid)
            bbox = f.getbbox('Ag')
            if bbox[3] - bbox[1] <= IMG_H - PADDING: lo = mid
            else: hi = mid - 1
        font_sizes[fp] = max(lo, 8)

        # Check cmap coverage
        try:
            tt = TTFont(fp, lazy=True)
            cmap = tt.getBestCmap() or {}
            supported = {chr(k) for k in cmap}
        except Exception:
            supported = set(IAM_ALPHABET)  # assume full coverage on error

        for c in IAM_ALPHABET:
            if c in supported:
                char_to_fonts[c].append(fp)

    # Report any char with no coverage
    for c, fps in char_to_fonts.items():
        if not fps:
            print(f'  WARNING: no font covers {repr(c)}, using all fonts as fallback')
            char_to_fonts[c] = list(font_paths)

    return char_to_fonts, font_sizes


def render_line(text, font_path, target_h=64,
                char_to_fonts=None, font_sizes=None):
    """Render text using PIL, char-by-char with fallback fonts for missing glyphs.

    If char_to_fonts is provided, each character that the primary font doesn't
    support is rendered with a randomly chosen fallback font that does support it.
    """
    from PIL import Image, ImageDraw, ImageFont

    PADDING = 4
    CHAR_PAD = 1  # horizontal padding between chars

    try:
        # Determine primary font size
        if font_sizes and font_path in font_sizes:
            primary_size = font_sizes[font_path]
        else:
            lo, hi = 8, target_h * 2
            for _ in range(20):
                mid = (lo + hi) // 2
                f = ImageFont.truetype(font_path, size=mid)
                bbox = f.getbbox('Ag')
                if bbox[3] - bbox[1] <= target_h - PADDING: lo = mid
                else: hi = mid - 1
            primary_size = max(lo, 8)

        primary_font = ImageFont.truetype(font_path, size=primary_size)

        # Determine which font has this primary path in char_to_fonts
        primary_supported = None
        if char_to_fonts is not None:
            primary_supported = {
                c for c, fps in char_to_fonts.items() if font_path in fps
            }

        # Render each character individually
        char_imgs = []
        for ch in text:
            # Choose font for this character
            if primary_supported is None or ch in primary_supported:
                use_font = primary_font
            else:
                fallbacks = [f for f in (char_to_fonts or {}).get(ch, [])
                             if f != font_path]
                if fallbacks:
                    fb = random.choice(fallbacks)
                    fb_size = font_sizes[fb] if font_sizes else primary_size
                    use_font = ImageFont.truetype(fb, size=fb_size)
                else:
                    use_font = primary_font

            bbox = use_font.getbbox(ch)
            cw = max(bbox[2] - bbox[0] + CHAR_PAD * 2, 4)
            cimg = Image.new('L', (cw, target_h), 255)
            d = ImageDraw.Draw(cimg)
            y = (target_h - (bbox[3] - bbox[1])) // 2 - bbox[1]
            d.text((CHAR_PAD - bbox[0], y), ch, font=use_font, fill=0)
            char_imgs.append(cimg)

        if not char_imgs:
            return None, None

        total_w = sum(c.width for c in char_imgs) + 8
        img_pil = Image.new('L', (total_w, target_h), 255)
        x = 4
        for ci in char_imgs:
            img_pil.paste(ci, (x, 0))
            x += ci.width

        img = np.array(img_pil)
        h, w = img.shape
        if h != target_h:
            new_w = max(1, int(w * target_h / h))
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        return img, text
    except Exception:
        return None, None


def render_line_split(text, font_path, target_h=64, max_w=2500,
                      char_to_fonts=None, font_sizes=None):
    """Render text, splitting at word boundaries if output exceeds max_w.
    Returns list of (img, label) pairs (usually just one, sometimes 2-3).
    """
    from PIL import ImageFont
    # Estimate char width using primary font size
    if font_sizes and font_path in font_sizes:
        fsize = font_sizes[font_path]
    else:
        fsize = 32
    try:
        f = ImageFont.truetype(font_path, size=fsize)
        avg_cw = f.getbbox('n')[2]  # rough char width
    except Exception:
        avg_cw = fsize * 0.6

    est_w = avg_cw * len(text) + 8
    if est_w <= max_w:
        img, lbl = render_line(text, font_path, target_h, char_to_fonts, font_sizes)
        return [(img, lbl)] if img is not None else []

    # Split at word boundaries into chunks
    words = text.split(' ')
    chunks, cur = [], []
    cur_est = 0
    for w in words:
        w_est = avg_cw * (len(w) + 1)
        if cur and cur_est + w_est > max_w:
            chunks.append(' '.join(cur))
            cur, cur_est = [w], w_est
        else:
            cur.append(w)
            cur_est += w_est
    if cur:
        chunks.append(' '.join(cur))

    results = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        img, lbl = render_line(chunk, font_path, target_h, char_to_fonts, font_sizes)
        if img is not None:
            results.append((img, lbl))
    return results


# ---------------------------------------------------------------------------
# LMDB writer
# ---------------------------------------------------------------------------
def write_lmdb(output_path, samples):
    os.makedirs(output_path, exist_ok=True)
    map_size = max(1 << 30, len(samples) * 80 * 1024)
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
# Character frequency report
# ---------------------------------------------------------------------------
def save_char_freq_report(samples, out_dir='./results'):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs(out_dir, exist_ok=True)
    all_text = ''.join(label for _, label in samples)
    counter = Counter(all_text)
    total = len(all_text)

    rows = []
    for c, n in sorted(counter.items(), key=lambda x: -x[1]):
        if c == ' ':    cat, label = 'Space',       'space'
        elif c.islower(): cat, label = 'Lowercase',  c
        elif c.isupper(): cat, label = 'Uppercase',  c
        elif c.isdigit(): cat, label = 'Digit',      c
        else:             cat, label = 'Punctuation', c
        rows.append((label, cat, n, round(n / total * 100, 3)))

    df = pd.DataFrame(rows, columns=['Character', 'Category', 'Count', 'Frequency(%)'])
    csv_path = os.path.join(out_dir, 'synth_char_freq.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved char freq CSV → {csv_path}')

    # Bar chart
    color_map = {'Lowercase':'#4C9BE8','Uppercase':'#F28C38',
                 'Digit':'#2ECC71','Space':'#9B59B6','Punctuation':'#E74C3C'}
    colors = [color_map[cat] for cat in df['Category']]
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(range(len(df)), df['Count'], color=colors, edgecolor='none', width=0.8)
    ax.set_yscale('log')
    ax.set_ylabel('Count (log scale)', fontsize=11)
    ax.set_title(
        f'Synthetic Training Set — Character Frequency (log scale)\n'
        f'{len(samples):,} lines | {total:,} chars | {len(df)} unique chars',
        fontsize=11)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Character'], fontsize=8)
    ax.axhline(500, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.text(len(df)-1, 550, '<500', ha='right', fontsize=8, color='gray')
    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    ax.legend(handles=patches, loc='upper right', fontsize=9)
    plt.tight_layout()
    png_path = os.path.join(out_dir, 'synth_char_freq.png')
    plt.savefig(png_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved char freq chart → {png_path}')

    # Summary stats
    print(f'\nSynthetic char distribution:')
    for cat in ['Lowercase','Uppercase','Digit','Space','Punctuation']:
        sub = df[df['Category']==cat]
        pct = sub['Count'].sum() / total * 100
        print(f'  {cat:12s}: {sub["Count"].sum():7,}  ({pct:.1f}%)')
    rare = df[df['Count'] < 500]
    print(f'\nChars still below 500: {len(rare)}')
    if len(rare):
        print('  ' + ', '.join(f'{r.Character}({r.Count})' for _, r in rare.iterrows()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_base',      type=int, default=15000,
                        help='Base number of lines to generate (before top-up)')
    parser.add_argument('--output_lmdb', default='./data/lmdb/train_synth')
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--rare_texts',  default='./data/rare_texts.txt',
                        help='Path to Doubao-generated rare texts (optional)')
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    fonts = get_fonts()
    print(f'Fonts available: {len(fonts)}')
    for f in fonts:
        print(f'  {os.path.basename(f)}')

    print('\nBuilding font coverage map...')
    char_to_fonts, font_sizes = build_font_coverage(fonts)

    print()
    pool = build_text_pool(args.rare_texts)

    print(f'\nRendering {args.n_base} synthetic images (max_w=2500)...')
    samples = []
    failed = 0
    i = 0
    while len(samples) < args.n_base:
        text      = pool[i % len(pool)]
        font_path = random.choice(fonts)
        i += 1

        pairs = render_line_split(text, font_path, max_w=2500,
                                  char_to_fonts=char_to_fonts,
                                  font_sizes=font_sizes)
        if not pairs:
            failed += 1
            continue

        for img, lbl in pairs:
            samples.append((img, lbl))
            if len(samples) >= args.n_base:
                break

        if len(samples) % 2000 == 0 and len(samples) > 0:
            print(f'  {len(samples)}/{args.n_base}  failed={failed}')

    print(f'\nGenerated: {len(samples)}  Failed: {failed}')

    write_lmdb(args.output_lmdb, samples)
    save_char_freq_report(samples, args.results_dir)


if __name__ == '__main__':
    main()
