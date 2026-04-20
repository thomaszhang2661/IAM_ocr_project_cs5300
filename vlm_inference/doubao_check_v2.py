"""
Use Doubao VLM to verify IAM handwriting annotations — v2 (high-precision prompt).

Key changes vs v1:
  - Bias reversed: default to CORRECT, only flag when certain
  - Context: IAM is professionally annotated, expect ~95%+ accuracy
  - Explicit false-positive traps listed (0/O confusion, abbreviations, etc.)
  - Confidence gate: must be able to spell out every changed character
  - AMBIGUOUS expanded: use when unsure, not just unreadable

Outputs saved to results/clean_2/ to avoid overwriting v1 results.

Usage:
    python vlm_inference/doubao_check_v2.py --split train --workers 20
    python vlm_inference/doubao_check_v2.py --split val   --workers 20
    python vlm_inference/doubao_check_v2.py --split test  --workers 20
"""

import argparse
import base64
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# ── Doubao defaults ───────────────────────────────────────────────────────────
DOUBAO_BASE  = 'https://ark.cn-beijing.volces.com/api/v3'
DOUBAO_KEY   = os.environ.get('DOUBAO_API_KEY', '')
DOUBAO_MODEL = os.environ.get('DOUBAO_MODEL_ID', 'ep-20260214152858-8r9sn')

# ── Prompt v2 ─────────────────────────────────────────────────────────────────
VERIFY_PROMPT_V2 = (
    'The image shows a single line of handwritten English text from the IAM '
    'Handwriting Database, which was annotated by trained human experts. '
    'The existing annotation is highly likely to be correct.\n\n'
    'The claimed annotation is:\n\n'
    '  "{annotation}"\n\n'
    'Your task: decide whether the annotation contains a clear, unambiguous error.\n\n'
    'Step 1 — Read the handwritten line carefully.\n'
    'Step 2 — Compare against the annotation word by word.\n'
    'Step 3 — Reply in EXACTLY one of these formats:\n\n'
    '  CORRECT\n'
    '  INCORRECT: <specific changes, e.g. "disc→dise"> | CORRECTED: <full corrected line>\n'
    '  AMBIGUOUS: <reason>\n\n'
    'CRITICAL RULES — read carefully:\n'
    '1. Default to CORRECT. If you are not highly confident there is an error, '
    'respond CORRECT.\n'
    '2. Use INCORRECT ONLY when ALL of the following are true:\n'
    '   - You can clearly read the handwritten word(s) in question\n'
    '   - The difference from the annotation is unambiguous (not a style variation)\n'
    '   - You can specify exactly which characters differ\n'
    '3. Use AMBIGUOUS when: the ink is faded/unclear, OR you can see a possible '
    'difference but are not certain enough to flag as INCORRECT.\n'
    '4. Do NOT flag these — they are NOT errors:\n'
    '   - Digit 0 vs letter O (IAM uses 0 for the digit zero; trust the annotation)\n'
    '   - Minor punctuation differences (commas, periods, hyphens, quotes, spacing)\n'
    '   - British spelling variants (colour, honour, etc.)\n'
    '   - Abbreviations and proper nouns (trust the annotation)\n'
    '   - Handwriting style variations that could be read either way\n'
    '5. Do not add any other text beyond the required format.'
)


def pil_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def verify_annotation(client: OpenAI, model_id: str, img, annotation: str,
                      retries: int = 3) -> str:
    img_b64 = pil_to_b64(img)
    prompt  = VERIFY_PROMPT_V2.format(annotation=annotation)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=150,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'image_url',
                         'image_url': {'url': f'data:image/png;base64,{img_b64}'}},
                        {'type': 'text', 'text': prompt},
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt * 5
            print(f'\n  [retry] Error (attempt {attempt+1}): {e}. Retrying in {wait}s...')
            time.sleep(wait)
    return 'ERROR'


def parse_verdict(response: str) -> tuple:
    """Returns (is_flagged, verdict_type, reason, corrected_text)."""
    resp_upper = response.strip().upper()

    if resp_upper.startswith('INCORRECT'):
        after = response.split(':', 1)[1].strip() if ':' in response else response
        corrected_text = ''
        reason = after
        if '| CORRECTED:' in after.upper():
            idx = after.upper().index('| CORRECTED:')
            reason = after[:idx].strip()
            corrected_text = after[idx + len('| CORRECTED:'):].strip()
        return True, 'INCORRECT', reason, corrected_text

    if resp_upper.startswith('AMBIGUOUS'):
        parts = response.split(':', 1)
        reason = parts[1].strip() if len(parts) > 1 else response
        return False, 'AMBIGUOUS', reason, ''

    if resp_upper.startswith('CORRECT'):
        return False, 'CORRECT', '', ''

    # Unexpected format — treat as AMBIGUOUS (conservative)
    return False, 'AMBIGUOUS', response, ''


def process_one(args_tuple):
    idx, sample, client, model_id, split = args_tuple
    gt       = sample['text']
    img      = sample['image']
    response = verify_annotation(client, model_id, img, gt)
    flagged, verdict_type, reason, corrected_text = parse_verdict(response)

    return {
        'idx':            idx,
        'split':          split,
        'ground_truth':   gt,
        'model_response': response,
        'verdict':        verdict_type,
        'flagged':        flagged,
        'reason':         reason,
        'corrected_text': corrected_text,
    }


def run(split, sample_n, output_dir, resume, client, model_id, workers):
    os.makedirs(output_dir, exist_ok=True)
    out_all     = os.path.join(output_dir, f'doubao_{split}_all.csv')
    out_flagged = os.path.join(output_dir, f'doubao_{split}_flagged.csv')
    out_summary = os.path.join(output_dir, f'doubao_{split}_summary.json')

    print(f'Loading Teklia/IAM-line ({split})...')
    hf_split = 'validation' if split == 'val' else split
    dataset  = load_dataset('Teklia/IAM-line', split=hf_split)
    print(f'  Total: {len(dataset)} samples')

    if sample_n and sample_n < len(dataset):
        import random
        random.seed(42)
        indices = random.sample(range(len(dataset)), sample_n)
        dataset = dataset.select(indices)
        print(f'  Subsampled to: {len(dataset)}')

    # Resume support
    done_ids: dict = {}
    if resume and os.path.exists(out_all):
        df_prev  = pd.read_csv(out_all)
        done_ids = {str(r['idx']): r for r in df_prev.to_dict('records')}
        print(f'  Resuming: {len(done_ids)} already done')

    todo = [(i, dataset[i]) for i in range(len(dataset)) if str(i) not in done_ids]
    print(f'  Remaining: {len(todo)}  (workers={workers})\n')

    lock         = threading.Lock()
    rows         = list(done_ids.values())
    save_counter = [0]

    tasks = [(idx, sample, client, model_id, split) for idx, sample in todo]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, t): t[0] for t in tasks}

        with tqdm(total=len(todo), desc=f'doubao_v2 ({split})') as pbar:
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as e:
                    idx = futures[future]
                    row = {
                        'idx': idx, 'split': split,
                        'ground_truth': '', 'model_response': 'ERROR',
                        'verdict': 'AMBIGUOUS', 'flagged': False,
                        'reason': str(e), 'corrected_text': '',
                    }

                with lock:
                    rows.append(row)
                    save_counter[0] += 1
                    if save_counter[0] % 100 == 0:
                        pd.DataFrame(rows).sort_values('idx').to_csv(out_all, index=False)

                pbar.update(1)

    df_all = pd.DataFrame(rows).sort_values('idx').reset_index(drop=True)
    df_all.to_csv(out_all, index=False)

    df_flagged = df_all[df_all['flagged']].copy()
    df_flagged.to_csv(out_flagged, index=False)

    n_total     = len(df_all)
    n_correct   = int((df_all['verdict'] == 'CORRECT').sum())
    n_ambiguous = int((df_all['verdict'] == 'AMBIGUOUS').sum())
    n_incorrect = int((df_all['verdict'] == 'INCORRECT').sum())

    summary = {
        'split':           split,
        'prompt_version':  'v2',
        'total_samples':   n_total,
        'correct_count':   n_correct,
        'ambiguous_count': n_ambiguous,
        'incorrect_count': n_incorrect,
        'correct_rate':    round(n_correct   / max(n_total, 1), 4),
        'ambiguous_rate':  round(n_ambiguous / max(n_total, 1), 4),
        'incorrect_rate':  round(n_incorrect / max(n_total, 1), 4),
    }
    with open(out_summary, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n=== Results (doubao_v2 / {split}) ===')
    print(f"Total:     {n_total}")
    print(f"CORRECT:   {n_correct}  ({summary['correct_rate']*100:.1f}%)")
    print(f"AMBIGUOUS: {n_ambiguous}  ({summary['ambiguous_rate']*100:.1f}%)")
    print(f"INCORRECT: {n_incorrect}  ({summary['incorrect_rate']*100:.1f}%)  ← flagged")
    print(f"\nFiles:\n  {out_all}\n  {out_flagged}\n  {out_summary}")

    return df_all, df_flagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',      default='test', choices=['train', 'test', 'val'])
    parser.add_argument('--sample',     type=int, default=None)
    parser.add_argument('--workers',    type=int, default=20)
    parser.add_argument('--output_dir', default='results/clean_2')
    parser.add_argument('--no_resume',  action='store_true')
    parser.add_argument('--api_key',    default=DOUBAO_KEY)
    parser.add_argument('--api_base',   default=DOUBAO_BASE)
    parser.add_argument('--model_id',   default=DOUBAO_MODEL)
    args = parser.parse_args()

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    run(
        split=args.split,
        sample_n=args.sample,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        client=client,
        model_id=args.model_id,
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
