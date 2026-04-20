"""
Confusion-table-based spell corrector.

For each word in the decoded output:
  1. If it's already in the vocabulary → keep
  2. Otherwise, apply one confusion operation (sub/merge/split) per position
  3. If any result lands in vocabulary → replace (prefer highest-count operation)
  4. If no single op works → try combining up to 2 ops (optional, slower)
"""

from __future__ import annotations
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


IAM_ALPHABET = (
    ' !"#&\'()*+,-./0123456789:;?'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abcdefghijklmnopqrstuvwxyz'
)

# Punctuation we strip before vocab lookup, then restore
_STRIP_CHARS = '.,!?;:"\'-()[]'


def _strip_punct(word: str) -> Tuple[str, str, str]:
    """Return (prefix_punct, core_word, suffix_punct)."""
    left  = len(word) - len(word.lstrip(_STRIP_CHARS))
    right = len(word) - len(word.rstrip(_STRIP_CHARS))
    if right == 0:
        return word[:left], word[left:], ''
    return word[:left], word[left:-right], word[-right:]


class ConfusionSpellCorrector:
    def __init__(
        self,
        confusion_csv: str,
        vocabulary: Set[str],
        min_count: int = 5,
        max_ops: int = 2,   # max ops per word (1=fast, 2=slower but better)
    ):
        self.vocab   = {w.lower() for w in vocabulary}
        self.max_ops = max_ops

        df = pd.read_csv(confusion_csv)
        df = df[df['count'] >= min_count].copy()

        # Fill NaN (insert/delete ops have one side empty)
        df['wrong']   = df['wrong'].fillna('')
        df['correct'] = df['correct'].fillna('')

        # Partition by operation type
        self.sub1: List[Tuple[str, str, int]] = []    # 1:1 substitutions
        self.merge: List[Tuple[str, str, int]] = []   # n:1 merges (wrong is longer)
        self.split: List[Tuple[str, str, int]] = []   # 1:n splits (correct is longer)
        # Skip pure insert/delete of spaces or punctuation-only — too noisy
        _skip_chars = set(' ' + _STRIP_CHARS)

        added_sub1: set = set()
        for _, row in df.iterrows():
            w, c, cnt = str(row['wrong']), str(row['correct']), int(row['count'])
            t = str(row['type'])
            if not w and not c:
                continue
            # Skip ops that are purely punctuation/space insertion or deletion
            if (not w or all(x in _skip_chars for x in w)) and \
               (not c or all(x in _skip_chars for x in c)):
                continue
            if t == 'substitute_1:1' and w and c:
                # Add both directions symmetrically
                for ww, cc in [(w, c), (c, w)]:
                    if (ww, cc) not in added_sub1:
                        self.sub1.append((ww, cc, cnt))
                        added_sub1.add((ww, cc))
            elif t == 'merge_n:1' and len(w) > 1 and len(c) == 1:
                self.merge.append((w, c, cnt))
                self.split.append((c, w, cnt))   # reverse: split
            elif t == 'split_1:n' and len(w) == 1 and len(c) > 1:
                self.split.append((w, c, cnt))
                self.merge.append((c, w, cnt))   # reverse: merge
            # replace_n:m: treat as combined op — skip for now (too many false positives)

        # Sort all by count descending (we prefer high-confidence corrections)
        self.sub1.sort(key=lambda x: -x[2])
        self.merge.sort(key=lambda x: -x[2])
        self.split.sort(key=lambda x: -x[2])

        print(f'ConfusionSpellCorrector: {len(self.sub1)} sub1, '
              f'{len(self.merge)} merge, {len(self.split)} split ops '
              f'(min_count={min_count})')

    # ------------------------------------------------------------------
    def _candidates_from_word(self, word: str) -> List[Tuple[str, int]]:
        """
        Generate all 1-op confusion corrections of `word`.
        Returns [(candidate_string, op_count), ...] sorted by count desc.
        """
        candidates: List[Tuple[str, int]] = []
        n = len(word)

        # 1:1 substitutions
        for wrong, correct, cnt in self.sub1:
            for i in range(n):
                if word[i] == wrong:
                    candidates.append((word[:i] + correct + word[i+1:], cnt))

        # merge: n chars → 1 char
        for wrong, correct, cnt in self.merge:
            wl = len(wrong)
            for i in range(n - wl + 1):
                if word[i:i+wl] == wrong:
                    candidates.append((word[:i] + correct + word[i+wl:], cnt))

        # split: 1 char → n chars
        for wrong, correct, cnt in self.split:
            for i in range(n):
                if word[i] == wrong:
                    candidates.append((word[:i] + correct + word[i+1:], cnt))

        return candidates

    def correct_word(self, word: str) -> str:
        """Return corrected word, or original if no correction found."""
        if not word:
            return word

        pre, core, suf = _strip_punct(word)
        if not core:
            return word

        core_lower = core.lower()
        if core_lower in self.vocab:
            return word  # already correct

        # Try single-op corrections
        cands = self._candidates_from_word(core_lower)
        # Only correct if there is a UNIQUE vocabulary match
        vocab_matches = [cand for cand, _ in cands if cand in self.vocab]
        unique = list(dict.fromkeys(vocab_matches))   # deduplicate, preserve order
        if len(unique) == 1:
            restored = _restore_case(core, unique[0])
            return pre + restored + suf

        # Try 2-op corrections (optional, only if max_ops=2)
        if self.max_ops >= 2:
            # For 2-op: only use high-confidence pairs (count >= 10) to avoid false positives
            high_conf_cands = [(cand, cnt) for cand, cnt in cands if cnt >= 10]
            seen = {core_lower}
            all_cands2 = []
            for cand1, _ in high_conf_cands:
                if cand1 in seen:
                    continue
                seen.add(cand1)
                for cand2, cnt2 in self._candidates_from_word(cand1):
                    if cand2 in self.vocab and cnt2 >= 10:
                        all_cands2.append(cand2)
            unique2 = list(dict.fromkeys(all_cands2))
            if len(unique2) == 1:
                restored = _restore_case(core, unique2[0])
                return pre + restored + suf

        return word  # no correction found

    def correct_line(self, text: str) -> str:
        """Apply word-level correction to a full decoded line."""
        tokens = text.split(' ')
        corrected = [self.correct_word(t) for t in tokens]
        return ' '.join(corrected)

    def correct_batch(self, texts: List[str]) -> List[str]:
        return [self.correct_line(t) for t in texts]


def _restore_case(original: str, corrected: str) -> str:
    """Restore capitalisation from original to corrected string."""
    if not original:
        return corrected
    if original[0].isupper() and corrected:
        return corrected[0].upper() + corrected[1:]
    if original.isupper():
        return corrected.upper()
    return corrected
