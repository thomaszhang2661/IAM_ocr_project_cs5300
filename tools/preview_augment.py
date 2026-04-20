"""
Preview augmentation: generate 10 augmented versions of training images.

Each augmented image has a label showing which ops were applied.

Usage:
    python tools/preview_augment.py
    python tools/preview_augment.py --idx 5 20 50 --n 10
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
from skimage import exposure
from skimage import filters as skfilters
from skimage import transform as skimage_tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Individual augmentation ops
# ---------------------------------------------------------------------------

def apply_affine(img_f32, tx=0.01, ty=0.01, sx=0.05, sy=0.05, shear=0.75):
    """
    Affine transform: scaling + shearing + translation.
    After transform, compute the bounding box of the four corners so that
    no content is clipped, then resize back to the original H.
    img_f32: H×W float32 in [0,1]
    """
    h, w = img_f32.shape

    _sx    = random.uniform(1. - sx, 1. + sx)
    _sy    = random.uniform(1. - sy, 1. + sy)
    _shear = random.uniform(-shear, shear)
    _tx    = random.uniform(-tx, tx) * w
    _ty    = random.uniform(-ty, ty) * h

    tf = skimage_tf.AffineTransform(
        scale=(_sx, _sy),
        shear=_shear,
        translation=(_tx, _ty),
    )

    # Transform the four corners to find the required output canvas size
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)
    new_corners = tf(corners)           # forward-map corners
    min_x, min_y = new_corners.min(axis=0)
    max_x, max_y = new_corners.max(axis=0)

    # Shift so the top-left lands at (0,0)
    shift = skimage_tf.AffineTransform(translation=(-min_x, -min_y))
    full_tf = shift + tf                # compose: first tf, then shift

    out_w = max(int(np.ceil(max_x - min_x)), 1)
    out_h = max(int(np.ceil(max_y - min_y)), 1)

    warped = skimage_tf.warp(
        img_f32, full_tf.inverse,
        output_shape=(out_h, out_w),
        cval=1.0,
    )

    # Resize back to original H, keep aspect ratio
    new_w = max(4, int(out_w * h / out_h))
    resized = cv2.resize(warped.astype(np.float32), (new_w, h),
                         interpolation=cv2.INTER_AREA)
    return resized


def apply_gamma(img_f32, gamma_max=1.0, gamma_min=0.5):
    """Gamma correction. gamma < 1 → brighter, gamma > 1 → darker."""
    gamma = random.uniform(gamma_min, gamma_max)
    return exposure.adjust_gamma(img_f32, gamma)


def apply_gaussian_noise(img_f32, std=0.03):
    """Add Gaussian noise (simulates scanner/camera noise)."""
    noise = np.random.normal(0, std, img_f32.shape).astype(np.float32)
    return np.clip(img_f32 + noise, 0.0, 1.0)


def apply_gaussian_blur(img_f32, sigma_max=1.0):
    """Gaussian blur (simulates defocus)."""
    sigma = random.uniform(0.3, sigma_max)
    return skfilters.gaussian(img_f32, sigma=sigma)


def apply_salt_pepper(img_f32, amount=0.01):
    """Salt & pepper noise (simulates paper spots / ink drops)."""
    out = img_f32.copy()
    n_salt   = int(amount * out.size * 0.5)
    n_pepper = int(amount * out.size * 0.5)
    # salt (white)
    coords = [np.random.randint(0, d, n_salt) for d in out.shape]
    out[coords[0], coords[1]] = 1.0
    # pepper (black)
    coords = [np.random.randint(0, d, n_pepper) for d in out.shape]
    out[coords[0], coords[1]] = 0.0
    return out


def apply_elastic(img_f32, alpha=8.0, sigma=3.0):
    """Elastic distortion (simulates pen tremor / ink bleeding)."""
    h, w = img_f32.shape
    dx = skfilters.gaussian(
        np.random.uniform(-1, 1, (h, w)).astype(np.float32), sigma=sigma
    ) * alpha
    dy = skfilters.gaussian(
        np.random.uniform(-1, 1, (h, w)).astype(np.float32), sigma=sigma
    ) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(img_f32, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Full augmentation pipeline
# ---------------------------------------------------------------------------

def full_augment(image_u8):
    """
    Apply all augmentations in sequence.
    image_u8: H×W uint8 grayscale
    Returns: H×W uint8
    """
    img = image_u8.astype(np.float32) / 255.0

    img = apply_affine(img)
    img = apply_gamma(img)
    img = apply_gaussian_blur(img)
    img = apply_gaussian_noise(img)
    img = apply_salt_pepper(img)
    img = apply_elastic(img)

    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx',        type=int, nargs='+', default=[5, 20, 50, 100, 200])
    parser.add_argument('--n',          type=int, default=10)
    parser.add_argument('--lmdb_dir',   default='./data/lmdb/train')
    parser.add_argument('--output_dir', default='./augment_preview')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import lmdb
    env = lmdb.open(args.lmdb_dir, readonly=True, lock=False)

    for sample_idx in args.idx:
        sample_dir = os.path.join(args.output_dir, f'sample_{sample_idx:04d}')
        os.makedirs(sample_dir, exist_ok=True)

        with env.begin() as txn:
            img_buf = txn.get(f'image-{sample_idx + 1:09d}'.encode())
            label   = txn.get(f'label-{sample_idx + 1:09d}'.encode()).decode('utf-8')

        img = cv2.imdecode(np.frombuffer(img_buf, np.uint8), cv2.IMREAD_GRAYSCALE)
        print(f'\n[idx={sample_idx}] "{label}"  shape={img.shape}')

        cv2.imwrite(os.path.join(sample_dir, '00_original.png'), img)

        for i in range(1, args.n + 1):
            aug = full_augment(img)
            cv2.imwrite(os.path.join(sample_dir, f'{i:02d}_aug.png'), aug)

        print(f'  → {args.n + 1} images saved to {sample_dir}/')

    env.close()
    total = len(args.idx) * (args.n + 1)
    print(f'\nDone. {total} images in {args.output_dir}/')


if __name__ == '__main__':
    main()
