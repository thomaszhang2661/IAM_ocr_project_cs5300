"""
Generate text sentences covering rare IAM characters using Doubao API.
Parallel version: concurrent API calls per character, saves incrementally.

Usage:
    python data/generate_rare_texts.py
    python data/generate_rare_texts.py --n_per_char 100 --workers 8
"""

import argparse
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

try:
    from secrets import DOUBAO_API_KEY, DOUBAO_MODEL_ID, DOUBAO_BASE_URL
    CLIENT = OpenAI(base_url=DOUBAO_BASE_URL, api_key=DOUBAO_API_KEY)
    MODEL  = DOUBAO_MODEL_ID
except ImportError:
    CLIENT = OpenAI(base_url='https://ark.cn-beijing.volces.com/api/v3', api_key='')
    MODEL  = 'ep-20260214152858-8r9sn'

IAM_ALPHABET = (
    ' !"#&\'()*+,-./0123456789:;?'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abcdefghijklmnopqrstuvwxyz'
)
VALID_CHARS = set(IAM_ALPHABET)

DEFAULT_RARE = list('0123456789ZQXKVUJYOBCDEFGHLNPRW?#&*+/;:()')


def make_prompt(char, n):
    if char.isdigit():
        hint = f"Include numbers, dates, measurements, statistics containing '{char}'."
    elif char.isupper():
        hint = f"Use proper nouns, acronyms, titles, abbreviations containing '{char}'."
    elif char in '?!':
        hint = "Use questions and exclamations."
    elif char in '&+*/':
        hint = f"Use math expressions, abbreviations, formulas with '{char}'."
    elif char in ';:':
        hint = "Use lists, time expressions, colons, semicolons."
    elif char in '()':
        hint = "Use parenthetical remarks, citations."
    elif char in '#/':
        hint = f"Use reference numbers, dates, fractions with '{char}'."
    else:
        hint = ""

    return f"""Generate exactly {n} short English sentences (one per line, no numbering, no blank lines).
Each sentence MUST contain the character '{char}' at least 2 times.
Realistic text: newspaper excerpts, book lines, addresses, measurements.
Length: 25-90 characters each.
{hint}
Only use: letters, digits, spaces, and common punctuation (.,!?;:'"()-&+*/#).
Output ONLY the sentences, nothing else."""


def fetch_for_char(char, n_target):
    """Fetch sentences for one character. Called in parallel."""
    results = []
    max_calls = 6
    for call_idx in range(max_calls):
        if len(results) >= n_target:
            break
        needed = min(n_target - len(results), 60)
        prompt = make_prompt(char, needed)
        try:
            resp = CLIENT.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=3000,
                temperature=0.88,
            )
            text = resp.choices[0].message.content.strip()
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                cleaned = ''.join(c for c in line if c in VALID_CHARS)
                if char in cleaned and len(cleaned) >= 15:
                    results.append(cleaned)
        except Exception as e:
            print(f"  [{char}] API error: {e}", flush=True)
            time.sleep(1)
    return char, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chars',      default=None)
    parser.add_argument('--n_per_char', type=int, default=100)
    parser.add_argument('--workers',    type=int, default=8,
                        help='Parallel API calls')
    parser.add_argument('--output',     default='./data/rare_texts.txt')
    args = parser.parse_args()

    chars = args.chars.split() if args.chars else DEFAULT_RARE
    chars = list(dict.fromkeys(chars))
    print(f'Target chars ({len(chars)}): {chars}', flush=True)
    print(f'n_per_char={args.n_per_char}  workers={args.workers}', flush=True)
    print(f'Output: {args.output}', flush=True)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    all_sentences = []
    char_counts: Counter = Counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_for_char, c, args.n_per_char): c for c in chars}
        for future in as_completed(futures):
            char, sentences = future.result()
            char_counts[char] = sum(char in s for s in sentences)
            all_sentences.extend(sentences)
            status = 'OK' if char_counts[char] >= args.n_per_char // 2 else f'LOW({char_counts[char]})'
            print(f"  ['{char}'] {len(sentences)} sentences  occurrences={char_counts[char]}  {status}", flush=True)
            # Incremental save
            with open(args.output, 'w', encoding='utf-8') as f:
                for s in all_sentences:
                    f.write(s + '\n')

    print(f'\n=== Done ===', flush=True)
    print(f'Total sentences: {len(all_sentences)}', flush=True)
    print(f'Saved → {args.output}', flush=True)

    low = [c for c in chars if char_counts.get(c, 0) < args.n_per_char // 2]
    if low:
        print(f'Low coverage chars: {low}', flush=True)


if __name__ == '__main__':
    main()
