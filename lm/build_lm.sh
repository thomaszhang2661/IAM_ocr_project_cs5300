#!/usr/bin/env bash
# Build word 4-gram KenLM from public English corpora + IAM train text.
# Outputs: lm/word_4gram.arpa  lm/word_4gram.binary
# All long steps run here so tmux keeps them alive after SSH disconnect.

set -e
CONDA_ENV=/data00/tiger/.local/share/conda/envs/ocr_IAM
PYBIN=$CONDA_ENV/bin/python3
LMPLZ=/tmp/kenlm/build/bin/lmplz
BUILD_BINARY=/tmp/kenlm/build/bin/build_binary
export LD_LIBRARY_PATH=$CONDA_ENV/lib:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LM_DIR="$SCRIPT_DIR"
mkdir -p "$LM_DIR"

echo "=== Step 1: collect IAM train text ==="
$PYBIN - <<'PYEOF'
import os, sys
sys.path.insert(0, os.path.expanduser('/data02/home/tiger/thomas/final_project'))

IAM_ALPHABET = set(
    ' !"#&\'()*+,-./0123456789:;?'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abcdefghijklmnopqrstuvwxyz'
)

lines = []
for csv_path in [
    './data/iam_hf/train/labels.csv',
    './data/iam_hf/validation/labels.csv',
]:
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        for t in df['text'].dropna():
            cleaned = ''.join(c for c in str(t) if c in IAM_ALPHABET)
            if cleaned.strip():
                lines.append(cleaned.strip())
    except Exception as e:
        print(f'  skip {csv_path}: {e}')

print(f'  IAM lines: {len(lines)}')
with open('./lm/corpus_iam.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')
PYEOF

echo "=== Step 2: download public English text (Project Gutenberg) ==="
$PYBIN - <<'PYEOF'
import urllib.request, re, os

IAM_ALPHABET = set(
    ' !"#&\'()*+,-./0123456789:;?'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abcdefghijklmnopqrstuvwxyz'
)

# A selection of public-domain English books from Project Gutenberg (plain text)
GUTENBERG_IDS = [
    1342,   # Pride and Prejudice
    11,     # Alice's Adventures in Wonderland
    84,     # Frankenstein
    1661,   # Sherlock Holmes
    2701,   # Moby Dick
    98,     # A Tale of Two Cities
    1400,   # Great Expectations
    2591,   # Grimm's Fairy Tales
    174,    # The Picture of Dorian Gray
    345,    # Dracula
    5200,   # Metamorphosis
    76,     # Adventures of Huckleberry Finn
    1232,   # The Prince
    2554,   # Crime and Punishment
    36,     # The War of the Worlds
    16,     # Peter Pan
    215,    # The Call of the Wild
    1260,   # Jane Eyre
    46,     # A Christmas Carol
    514,    # Little Women
]

all_lines = []
for gid in GUTENBERG_IDS:
    url = f'https://www.gutenberg.org/files/{gid}/{gid}-0.txt'
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read().decode('utf-8', errors='ignore')
    except Exception:
        # Try alternate URL format
        try:
            url2 = f'https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt'
            with urllib.request.urlopen(url2, timeout=30) as r:
                raw = r.read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f'  skip {gid}: {e}')
            continue

    # Strip Gutenberg header/footer
    start = re.search(r'\*\*\* START OF (THE|THIS) PROJECT GUTENBERG', raw)
    end   = re.search(r'\*\*\* END OF (THE|THIS) PROJECT GUTENBERG', raw)
    if start:
        raw = raw[start.end():]
    if end:
        raw = raw[:end.start()]

    for line in raw.splitlines():
        line = line.strip()
        if not line or len(line) < 20:
            continue
        cleaned = ''.join(c for c in line if c in IAM_ALPHABET)
        if len(cleaned) >= 20:
            all_lines.append(cleaned)

    print(f'  book {gid}: {len(all_lines)} lines so far', flush=True)

print(f'\nTotal public corpus lines: {len(all_lines)}')
with open('./lm/corpus_public.txt', 'w') as f:
    f.write('\n'.join(all_lines) + '\n')
PYEOF

echo "=== Step 3: merge corpora ==="
cat lm/corpus_iam.txt lm/corpus_public.txt > lm/corpus_all.txt
wc -l lm/corpus_all.txt

echo "=== Step 4: train 4-gram KenLM (word-level) ==="
# lmplz reads sentences from stdin (one per line), outputs ARPA to stdout
$LMPLZ -o 4 \
  --discount_fallback \
  -S 20% -T /tmp/kenlm_tmp \
  < lm/corpus_all.txt \
  > lm/word_4gram.arpa

echo "=== Step 5: binarize ==="
$BUILD_BINARY lm/word_4gram.arpa lm/word_4gram.binary

echo ""
echo "=== Done ==="
ls -lh lm/word_4gram.arpa lm/word_4gram.binary
