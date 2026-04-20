"""
LMDB dataset class for CRNN training on IAM.
Compatible with meijieru/crnn.pytorch LMDB format.
"""

import os
import sys

import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class LMDBDataset(Dataset):
    """
    Dataset reading from LMDB created by prepare_lmdb.py.

    Each item is stored as:
        image-{idx:09d}  -> PNG bytes
        label-{idx:09d}  -> text (utf-8)
    """
    def __init__(self, lmdb_path, img_h=64, img_w=None, transform=None, augment=False):
        """
        Args:
            lmdb_path: path to LMDB directory
            img_h: target image height (must be 32 for CRNN)
            img_w: if set, resize width to this value; otherwise use original width
            transform: torchvision transform; if None, uses default normalization
        """
        self.env = lmdb.open(
            lmdb_path,
            max_readers=8,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin() as txn:
            self.n_samples = int(txn.get(b'num-samples').decode())

        self.img_h   = img_h
        self.img_w   = img_w
        self.augment = augment

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        idx = index + 1  # LMDB keys are 1-indexed
        with self.env.begin() as txn:
            img_key = f'image-{idx:09d}'.encode()
            lbl_key = f'label-{idx:09d}'.encode()

            img_buf  = txn.get(img_key)
            lbl_buf  = txn.get(lbl_key)

        # Decode image
        img_arr = np.frombuffer(img_buf, dtype=np.uint8)
        import cv2
        img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)

        # Resize if needed
        h, w = img.shape
        if self.img_w:
            img = cv2.resize(img, (self.img_w, self.img_h))
        elif h != self.img_h:
            new_w = max(1, int(w * self.img_h / h))
            img = cv2.resize(img, (new_w, self.img_h))

        # Online augmentation (training only)
        if self.augment:
            from htr_model.augment import augment as aug_fn
            img = aug_fn(img)

        img_pil = Image.fromarray(img)
        img_tensor = self.transform(img_pil)  # (1, H, W)

        label = lbl_buf.decode('utf-8')
        return img_tensor, label


def collate_fn(batch):
    """
    Custom collate: pad images to same width, keep labels as list.
    Images: (N, C, H, W_max)
    Labels: list of strings
    """
    images, labels = zip(*batch)

    # Find max width
    max_w = max(img.shape[-1] for img in images)

    # Pad to max width with -1 (normalized background ≈ white)
    padded = []
    for img in images:
        pad_w = max_w - img.shape[-1]
        padded.append(
            torch.nn.functional.pad(img, (0, pad_w), value=1.0)   # white background
        )

    return torch.stack(padded, 0), list(labels)


class Converter:
    """
    Encode labels to integer sequences and decode integer sequences back to strings.
    Blank token is index 0.
    """
    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        # char -> index (1-indexed; 0 is CTC blank)
        self.char2idx = {c: i + 1 for i, c in enumerate(alphabet)}
        # index -> char
        self.idx2char = {i + 1: c for i, c in enumerate(alphabet)}

    def encode(self, texts):
        """
        Encode list of strings to (targets, target_lengths).
        targets: flat LongTensor
        target_lengths: LongTensor of length N
        """
        all_idx = []
        lengths = []
        for t in texts:
            encoded = []
            for c in t:
                if c in self.char2idx:
                    encoded.append(self.char2idx[c])
                # silently skip unknown chars
            all_idx.extend(encoded)
            lengths.append(len(encoded))
        return (
            torch.LongTensor(all_idx),
            torch.LongTensor(lengths),
        )

    def decode(self, indices, lengths=None):
        """
        Decode CTC output (greedy, merge repeated and remove blanks).
        indices: flat list or LongTensor of indices
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        # Greedy decode: merge consecutive identical, remove blanks
        result = []
        prev = None
        for idx in indices:
            if idx != prev and idx != 0:
                result.append(self.idx2char.get(idx, ''))
            prev = idx
        return ''.join(result)

    def decode_batch(self, log_probs):
        """
        Greedy decode from log_probs tensor.
        log_probs: (T, N, num_classes)
        Returns list of N decoded strings.
        """
        # Argmax over class dimension
        preds = log_probs.argmax(dim=2)  # (T, N)
        preds = preds.permute(1, 0)      # (N, T)
        decoded = []
        for seq in preds:
            decoded.append(self.decode(seq.tolist()))
        return decoded
