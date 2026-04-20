"""
IAM Handwriting Dataset Loader

IAM directory structure expected:
    iam_data/
        ascii/
            lines.txt
        lines/
            a01/
                a01-000u/
                    a01-000u-00.png
                    ...
        task1/
            trainset.txt
            testset.txt
            validationset1.txt
            validationset2.txt
"""

import os
from pathlib import Path


def parse_lines_txt(lines_txt_path):
    """
    Parse IAM ascii/lines.txt.
    Returns dict: line_id -> {'status': str, 'text': str}

    Line format:
        a01-000u-00 ok 154 19 408 746 150 183 A MOVE to stop Mr.
        fields: id status graylevel components x y w h transcription...
    """
    records = {}
    with open(lines_txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(' ')
            if len(parts) < 9:
                continue
            line_id = parts[0]
            status = parts[1]
            text = ' '.join(parts[8:])
            records[line_id] = {'status': status, 'text': text}
    return records


def load_split_ids(split_file):
    """Load line IDs from a split file (trainset.txt etc.)."""
    ids = set()
    with open(split_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def get_image_path(iam_root, line_id):
    """
    Build image path from line ID.
    e.g. a01-000u-00 -> lines/a01/a01-000u/a01-000u-00.png
    """
    parts = line_id.split('-')
    writer = parts[0]               # a01
    form = f'{parts[0]}-{parts[1]}'  # a01-000u
    filename = f'{line_id}.png'
    return os.path.join(iam_root, 'lines', writer, form, filename)


def load_split(iam_root, split='test', ok_only=True):
    """
    Load all samples for a given split.

    Args:
        iam_root: path to IAM root directory
        split: 'train', 'test', 'val1', or 'val2'
        ok_only: only include lines with status='ok'

    Returns:
        list of dicts: [{'id', 'text', 'image_path'}, ...]
    """
    split_map = {
        'train': 'trainset.txt',
        'test': 'testset.txt',
        'val1': 'validationset1.txt',
        'val2': 'validationset2.txt',
    }
    if split not in split_map:
        raise ValueError(f"split must be one of {list(split_map.keys())}")

    task_dir = os.path.join(iam_root, 'task1')
    split_file = os.path.join(task_dir, split_map[split])
    if not os.path.exists(split_file):
        # Some IAM releases use a different split directory name
        task_dir = os.path.join(iam_root, 'splits')
        split_file = os.path.join(task_dir, split_map[split])

    split_ids = load_split_ids(split_file)

    lines_txt = os.path.join(iam_root, 'ascii', 'lines.txt')
    records = parse_lines_txt(lines_txt)

    samples = []
    for line_id in sorted(split_ids):
        if line_id not in records:
            continue
        rec = records[line_id]
        if ok_only and rec['status'] != 'ok':
            continue
        img_path = get_image_path(iam_root, line_id)
        if not os.path.exists(img_path):
            continue
        samples.append({
            'id': line_id,
            'text': rec['text'],
            'image_path': img_path,
            'writer': line_id.split('-')[0],
        })
    return samples


def get_iam_stats(iam_root):
    """Print dataset statistics."""
    all_records = parse_lines_txt(os.path.join(iam_root, 'ascii', 'lines.txt'))
    splits = {s: load_split(iam_root, s) for s in ['train', 'test', 'val1', 'val2']}
    print("IAM Dataset Statistics")
    print("=" * 40)
    print(f"Total records in lines.txt: {len(all_records)}")
    for name, samples in splits.items():
        writers = set(s['writer'] for s in samples)
        print(f"  {name:6s}: {len(samples):5d} lines, {len(writers):3d} writers")
    return splits


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python iam_loader.py <iam_root>")
        sys.exit(1)
    get_iam_stats(sys.argv[1])
