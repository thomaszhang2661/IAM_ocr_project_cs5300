"""
Use a VLM to verify IAM handwriting annotations (concurrent version).

Strategy: send image + ground-truth annotation together and ask the model
"Is this annotation correct?" — the model directly judges quality and explains
any errors.

Flagging rule: CORRECT / INCORRECT / AMBIGUOUS

Usage:
    python vlm_inference/doubao_check.py --split test --workers 20
    python vlm_inference/doubao_check.py --split train --workers 20
    python vlm_inference/doubao_check.py --split test --sample 200 --workers 10
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

# ── Doubao defaults ──────────────────────────────────────────────────────────
DOUBAO_BASE  = 'https://ark.cn-beijing.volces.com/api/v3'
DOUBAO_KEY   = os.environ.get('DOUBAO_API_KEY', '')
DOUBAO_MODEL = os.environ.get('DOUBAO_MODEL_ID', 'ep-20260214152858-8r9sn')

# ── Prompt ───────────────────────────────────────────────────────────────────
VERIFY_PROMPT = (
    'The image shows a single line of handwritten English text.\n'
    'The claimed annotation for this line is:\n\n'
    '  "{annotation}"\n\n'
    'Step 1 — Read every word in the image carefully.\n'
    'Step 2 — Compare word by word against the annotation.\n'
    'Step 3 — Reply in exactly one of these three formats:\n\n'
    '  CORRECT\n'
    '  INCORRECT: <word changes, e.g. "disc→dise"> | CORRECTED: <full corrected line>\n'
    '  AMBIGUOUS: <specific words whose ink is too faded/blurry to read at all>\n\n'
    'Rules:\n'
    '- Ignore punctuation and spacing differences (commas, periods, hyphens, quotes).\n'
    '- INCORRECT: use this whenever you can read a word and it differs from the annotation.\n'
    '  Err on the side of flagging.\n'
    '- AMBIGUOUS: use this ONLY when the ink is so faded or blurry that you literally\n'
    '  cannot make out the letters — not for uncertainty about correctness.\n'
    '- Do not add any other text.'
)


def pil_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def verify_annotation(client: OpenAI, model_id: str, img, annotation: str,
                      retries: int = 3) -> str:
    img_b64 = pil_to_b64(img)
    prompt  = VERIFY_PROMPT.format(annotation=annotation)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=128,
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
            print(f'\n  [idx] Error (attempt {attempt+1}): {e}. Retrying in {wait}s...')
            time.sleep(wait)
    return 'ERROR'


def parse_verdict(response: str) -> tuple:
    """Returns (is_incorrect, verdict_type, reason, corrected_text)."""
    resp = response.strip().upper()

    if resp.startswith('INCORRECT'):
        after = response.split(':', 1)[1].strip() if ':' in response else response
        corrected_text = ''
        reason = after
        if '| CORRECTED:' in after:
            parts = after.split('| CORRECTED:', 1)
            reason = parts[0].strip()
            corrected_text = parts[1].strip()
        return True, 'INCORRECT', reason, corrected_text

    if resp.startswith('AMBIGUOUS'):
        parts = response.split(':', 1)
        reason = parts[1].strip() if len(parts) > 1 else response
        return False, 'AMBIGUOUS', reason, ''

    if resp.startswith('CORRECT'):
        return False, 'CORRECT', '', ''

    # Unexpected format — treat as ambiguous
    return False, 'AMBIGUOUS', response, ''


def process_one(args_tuple):
    """Worker function — called in thread pool."""
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

    # ── Load dataset ──
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

    # ── Resume ──
    done_ids: dict = {}  # idx -> row dict
    if resume and os.path.exists(out_all):
        df_prev  = pd.read_csv(out_all)
        done_ids = {str(r['idx']): r for r in df_prev.to_dict('records')}
        print(f'  Resuming: {len(done_ids)} already done')

    todo = [(i, dataset[i]) for i in range(len(dataset)) if str(i) not in done_ids]
    print(f'  Remaining: {len(todo)}  (workers={workers})\n')

    # ── Concurrent inference ──
    lock   = threading.Lock()
    rows   = list(done_ids.values())  # start with already-done rows
    save_counter = [0]  # mutable counter for periodic saves

    tasks = [(idx, sample, client, model_id, split) for idx, sample in todo]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, t): t[0] for t in tasks}

        with tqdm(total=len(todo), desc=f'doubao ({split})') as pbar:
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as e:
                    idx = futures[future]
                    print(f'\n  Worker error idx={idx}: {e}')
                    row = {
                        'idx': idx, 'split': split,
                        'ground_truth': '', 'model_response': 'ERROR',
                        'verdict': 'AMBIGUOUS', 'flagged': False,
                        'reason': str(e), 'corrected_text': '',
                    }

                with lock:
                    rows.append(row)
                    save_counter[0] += 1
                    if save_counter[0] % 50 == 0:
                        pd.DataFrame(rows).sort_values('idx').to_csv(out_all, index=False)

                pbar.update(1)

    # ── Final save ──
    df_all = pd.DataFrame(rows).sort_values('idx').reset_index(drop=True)
    df_all.to_csv(out_all, index=False)

    df_flagged = df_all[df_all['flagged']].copy()
    df_flagged.to_csv(out_flagged, index=False)

    # ── Summary ──
    n_total     = len(df_all)
    n_correct   = int((df_all['verdict'] == 'CORRECT').sum())
    n_ambiguous = int((df_all['verdict'] == 'AMBIGUOUS').sum())
    n_incorrect = int((df_all['verdict'] == 'INCORRECT').sum())

    summary = {
        'split':            split,
        'total_samples':    n_total,
        'correct_count':    n_correct,
        'ambiguous_count':  n_ambiguous,
        'incorrect_count':  n_incorrect,
        'correct_rate':     round(n_correct   / max(n_total, 1), 4),
        'ambiguous_rate':   round(n_ambiguous / max(n_total, 1), 4),
        'incorrect_rate':   round(n_incorrect / max(n_total, 1), 4),
    }
    with open(out_summary, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n=== Results (doubao / {split}) ===')
    print(f"Total:     {n_total}")
    print(f"CORRECT:   {n_correct}    ({summary['correct_rate']*100:.1f}%)")
    print(f"AMBIGUOUS: {n_ambiguous}  ({summary['ambiguous_rate']*100:.1f}%)")
    print(f"INCORRECT: {n_incorrect}  ({summary['incorrect_rate']*100:.1f}%)  ← flagged")
    print(f"\nFiles:\n  {out_all}\n  {out_flagged}\n  {out_summary}")

    return df_all, df_flagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',    default='test', choices=['train', 'test', 'val'])
    parser.add_argument('--sample',   type=int, default=None,
                        help='Subsample N lines (default: all)')
    parser.add_argument('--workers',  type=int, default=20,
                        help='Number of concurrent API workers (default: 20)')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--no_resume',  action='store_true')
    parser.add_argument('--api_key',   default=DOUBAO_KEY)
    parser.add_argument('--api_base',  default=DOUBAO_BASE)
    parser.add_argument('--model_id',  default=DOUBAO_MODEL)
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
