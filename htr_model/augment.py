"""
Online image augmentation for HTR training.
Based on tools/image_utils.py (recognition_augment_transform) + noise ops.
All functions operate on uint8 grayscale numpy arrays.
"""

import random

import cv2
import numpy as np
from skimage import exposure
from skimage import filters as skfilters
from skimage import transform as skimage_tf


def _to_f32(img):
    """uint8 [0,255] → float32 [0,1]"""
    return img.astype(np.float32) / 255.0


def _to_u8(img):
    """float32 [0,1] → uint8 [0,255]"""
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def apply_affine(img_f32, tx=0.01, ty=0.01, sx=0.05, sy=0.05, shear=0.75):
    """
    Affine: scaling + shearing + translation.
    After transform, compute the bounding box of the four corners so that
    no content is clipped, then resize back to original H.
    Reference: tools/image_utils.py recognition_augment_transform
    """
    h, w = img_f32.shape
    _sx    = random.uniform(1. - sx, 1. + sx)
    _sy    = random.uniform(1. - sy, 1. + sy)
    _shear = random.uniform(-shear, shear)
    _tx    = random.uniform(-tx, tx) * w
    _ty    = random.uniform(-ty, ty) * h

    tf = skimage_tf.AffineTransform(
        scale=(_sx, _sy), shear=_shear, translation=(_tx, _ty)
    )

    # Forward-map four corners → bounding box
    corners     = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)
    new_corners = tf(corners)
    min_x, min_y = new_corners.min(axis=0)
    max_x, max_y = new_corners.max(axis=0)

    # Shift so top-left → (0, 0)
    shift   = skimage_tf.AffineTransform(translation=(-min_x, -min_y))
    full_tf = shift + tf

    out_w = max(4, int(np.ceil(max_x - min_x)))
    out_h = max(4, int(np.ceil(max_y - min_y)))

    warped = skimage_tf.warp(img_f32, full_tf.inverse,
                             output_shape=(out_h, out_w), cval=1.0)

    # Resize back to original H, keep aspect ratio
    new_w = max(4, int(out_w * h / out_h))
    return cv2.resize(warped.astype(np.float32), (new_w, h),
                      interpolation=cv2.INTER_AREA)


def apply_gamma(img_f32, lo=0.7, hi=2.5):
    """Gamma correction. <1 brightens, >1 darkens.
    Adaptive: if image is already light (mean > 0.85), force gamma >= 1
    to avoid washing out faint ink further. Otherwise allow full range.
    """
    mean_brightness = img_f32.mean()
    if mean_brightness > 0.85:   # light image → only darken
        gamma = random.uniform(1.0, hi)
    else:                         # dark/medium image → full range
        gamma = random.uniform(lo, hi)
    return exposure.adjust_gamma(img_f32, gamma)


def apply_gaussian_blur(img_f32, sigma_max=1.0):
    """Gaussian blur — simulates defocus / ink spread."""
    sigma = random.uniform(0.3, sigma_max)
    return skfilters.gaussian(img_f32, sigma=sigma)


def apply_gaussian_noise(img_f32, std=0.03):
    """Additive Gaussian noise — simulates scanner/camera noise."""
    noise = np.random.normal(0, std, img_f32.shape).astype(np.float32)
    return np.clip(img_f32 + noise, 0.0, 1.0)


def apply_salt_pepper(img_f32, amount=0.008):
    """Salt & pepper noise — simulates paper spots / ink drops."""
    out = img_f32.copy()
    n = int(amount * out.size)
    # salt (white)
    ri = np.random.randint(0, out.shape[0], n)
    ci = np.random.randint(0, out.shape[1], n)
    out[ri, ci] = 1.0
    # pepper (black)
    ri = np.random.randint(0, out.shape[0], n)
    ci = np.random.randint(0, out.shape[1], n)
    out[ri, ci] = 0.0
    return out


def apply_elastic(img_f32, alpha=6.0, sigma=3.0):
    """Elastic distortion — simulates pen tremor."""
    h, w = img_f32.shape
    dx = skfilters.gaussian(
        np.random.uniform(-1, 1, (h, w)).astype(np.float32), sigma=sigma
    ) * alpha
    dy = skfilters.gaussian(
        np.random.uniform(-1, 1, (h, w)).astype(np.float32), sigma=sigma
    ) * alpha
    x, y   = np.meshgrid(np.arange(w), np.arange(h))
    map_x  = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y  = np.clip(y + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(img_f32, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def augment(image_u8,
            tx=0.01, ty=0.01, sx=0.05, sy=0.05, shear=0.75,
            gamma_lo=0.001, gamma_hi=1.0,
            blur_sigma=1.0, noise_std=0.03,
            sp_amount=0.008, elastic_alpha=6.0):
    """
    Full augmentation pipeline (all ops applied in sequence).
    image_u8: H×W uint8 grayscale
    Returns: H×W uint8
    """
    img = _to_f32(image_u8)
    img = apply_affine(img, tx=tx, ty=ty, sx=sx, sy=sy, shear=shear)
    img = apply_gamma(img, lo=gamma_lo, hi=gamma_hi)
    img = apply_gaussian_blur(img, sigma_max=blur_sigma)
    img = apply_gaussian_noise(img, std=noise_std)
    img = apply_salt_pepper(img, amount=sp_amount)
    img = apply_elastic(img, alpha=elastic_alpha)
    return _to_u8(img)
