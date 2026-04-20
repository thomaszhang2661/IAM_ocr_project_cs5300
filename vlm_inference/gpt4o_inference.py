"""
Run GPT-4o on IAM line images for transcription.

Usage:
    python gpt4o_inference.py --iam_root /path/to/iam \
                               --split test \
                               --output results/gpt4o_test.json \
                               --sample 500
"""

import argparse
import base64
import json
import os
import sys
import time

from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.iam_loader import load_split

MODEL = 'gpt-4o'

SYSTEM_PROMPT = (
    "You are a handwriting transcription assistant. "
    "Your task is to read handwritten text from an image and output the exact transcription. "
    "Output ONLY the transcribed text, with no explanation, punctuation changes, or formatting. "
    "Preserve the original capitalization, punctuation, and spacing as written."
)

USER_PROMPT = (
    "Please transcribe the handwritten text in this image exactly as written. "
    "Output only the transcription, nothing else."
)


def encode_image_b64(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')


def transcribe_image(client: OpenAI, image_path: str, retries: int = 3) -> str:
    """Call GPT-4o API to transcribe a single image."""
    img_b64 = encode_image_b64(image_path)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=256,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f'data:image/png;base64,{img_b64}',
                                    'detail': 'high',
                                },
                            },
                            {'type': 'text', 'text': USER_PROMPT},
                        ],
                    },
                ],
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            wait = 2 ** attempt * 15
            print(f"\n  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except APIError as e:
            print(f"\n  API error on attempt {attempt+1}: {e}")
            if attempt == retries - 1:
                return ''
            time.sleep(5)

    return ''


def run_inference(iam_root, split, output_path, sample_n=None, resume=True):
    client = OpenAI()  # reads OPENAI_API_KEY from env

    samples = load_split(iam_root, split)
    print(f"Loaded {len(samples)} samples from split='{split}'")

    if sample_n and sample_n < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, sample_n)
        print(f"Subsampled to {len(samples)}")

    results = {}
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resuming: already have {len(results)} results")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    todo = [s for s in samples if s['id'] not in results]
    print(f"Remaining: {len(todo)} images to process")

    for i, sample in enumerate(tqdm(todo, desc='GPT-4o inference')):
        pred = transcribe_image(client, sample['image_path'])
        results[sample['id']] = {
            'ground_truth': sample['text'],
            'gpt4o_pred': pred,
            'image_path': sample['image_path'],
            'writer': sample['writer'],
        }

        if (i + 1) % 50 == 0:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iam_root', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'test', 'val1', 'val2'])
    parser.add_argument('--output', default='results/gpt4o_test.json')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    args = parser.parse_args()

    run_inference(
        iam_root=args.iam_root,
        split=args.split,
        output_path=args.output,
        sample_n=args.sample,
        resume=not args.no_resume,
    )


if __name__ == '__main__':
    main()
