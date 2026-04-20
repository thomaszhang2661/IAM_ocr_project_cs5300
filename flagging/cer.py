"""
Character Error Rate (CER) utilities.

CER = edit_distance(hypothesis, reference) / len(reference)

Uses character-level edit distance (not word-level WER).
"""

import editdistance


def compute_cer(hypothesis: str, reference: str, normalize: bool = True) -> float:
    """
    Compute CER between hypothesis and reference strings.

    Args:
        hypothesis: predicted/transcribed text
        reference: ground-truth text
        normalize: if True, divide by len(reference). If reference is empty,
                   returns 0.0 when hypothesis is also empty, else 1.0.

    Returns:
        CER as a float in [0, inf) (normalized) or int (unnormalized)
    """
    # Normalize whitespace
    hyp = ' '.join(hypothesis.strip().split())
    ref = ' '.join(reference.strip().split())

    dist = editdistance.eval(hyp, ref)

    if not normalize:
        return dist

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    return dist / len(ref)


def compute_cer_batch(hypotheses, references):
    """
    Compute CER for a list of (hypothesis, reference) pairs.
    Returns list of CER values.
    """
    return [compute_cer(h, r) for h, r in zip(hypotheses, references)]


def aggregate_cer(hypotheses, references):
    """
    Compute corpus-level CER (total edit distance / total reference length).
    This is the standard way HTR papers report CER.
    """
    total_dist = 0
    total_len = 0
    for hyp, ref in zip(hypotheses, references):
        hyp = ' '.join(hyp.strip().split())
        ref = ' '.join(ref.strip().split())
        total_dist += editdistance.eval(hyp, ref)
        total_len += len(ref)
    if total_len == 0:
        return 0.0
    return total_dist / total_len


def normalize_text(text: str) -> str:
    """Basic text normalization: strip and collapse whitespace."""
    return ' '.join(text.strip().split())
