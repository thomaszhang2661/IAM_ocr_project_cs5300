#!/usr/bin/env python
# encoding: utf-8
# author: Guanghui Ren
# license: (C) Copyright 2019, YITU INC Limited.
# contact: guanghui.ren@yitu-inc.com
# file: image_utils.py
# time: 9/1/19 11:22 AM
# desc:
import cv2
from skimage import transform as skimage_tf
from skimage import exposure
import random
import torch
import numpy as np
import numpy as np
from skimage import io
import cv2
import base64


def recognition_augment_transform(image, random_x_translation, random_y_translation, random_x_scaling, random_y_scaling,
                                  random_shearing, random_gamma):
    '''
    This function randomly:
        - translates the input image by +-width_range and +-height_range (percentage).
        - scales the image by y_scaling and x_scaling (percentage)
        - shears the image by shearing_factor (radians)
    '''

    ty = random.uniform(-random_y_translation, random_y_translation)
    tx = random.uniform(-random_x_translation, random_x_translation)

    sx = random.uniform(1. - random_y_scaling, 1. + random_y_scaling)
    sy = random.uniform(1. - random_x_scaling, 1. + random_x_scaling)

    s = random.uniform(-random_shearing, random_shearing)
    gamma = random.uniform(0.001, random_gamma)
    image = exposure.adjust_gamma(image, gamma)

    st = skimage_tf.AffineTransform(scale=(sx, sy),
                                    shear=s,
                                    translation=(tx * image.shape[1], ty * image.shape[0]))
    augmented_image = skimage_tf.warp(image, st, cval=1.0)
    return augmented_image*255
def recognition_transform(image, label, mean, std, alphabet_dict, max_seq_len):
    image = np.expand_dims(image, axis=0).astype(np.float32)
    if np.max(image) > 1:
        image = image / 255.
    image = (image - mean) / std
    image = torch.from_numpy(image.copy())
    label_encode = np.zeros(max_seq_len, dtype=np.float32) - 1
    i = 0
    for letter in label:
        label_encode[i] = alphabet_dict[letter]
        i += 1
    label_encode = torch.from_numpy(label_encode).long()
    label_len = torch.from_numpy(np.array([i])).long()
    return image, label_encode, label_len

def crop_image_with_jitter(image, x1, y1, x2, y2, random_jitter):
    assert random_jitter < 1
    h = y2-y1+1
    delta = int(h * np.random.uniform(-random_jitter, random_jitter))
    y1 = max(0, y1+delta)
    y2 = min(y2+delta,image.shape[0]-1)
    image = image[y1:y2 + 1, x1:x2 + 1]
    return image

# def crop_image2(image, x1, y1, x2, y2):
#     X2 = min(x2 + 1, image.shape[1])
#     Y2 = min(y2+1, image.shape[0])
#     image = image[y1:y2 + 1, x1:x2 + 1]
#     return image

def crop_image(image, bb):
    ''' Helper function to crop the image by the bounding box (in percentages)
    '''
    (x, y, w, h) = bb
    x = x * image.shape[1]
    y = y * image.shape[0]
    w = w * image.shape[1]
    h = h * image.shape[0]
    (x1, y1, x2, y2) = (x, y, x + w, y + h)
    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
    return image[y1:y2, x1:x2]


def resize_image(image, desired_size):
    ''' Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------

    image: np.array
        The image to be resized.

    desired_size: (int, int)
        The (height, width) of the resized image

    Return
    ------

    image: np.array
        The image of size = desired_size

    bounding box: (int, int, int, int)
        (x, y, w, h) in percentages of the resized image of the original
    '''
    size = image.shape[:2]
    # print('size:', size)
    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0]) / size[0]
        ratio_h = float(desired_size[1]) / size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x * ratio) for x in size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    # top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    # left, right = delta_w // 2, delta_w - (delta_w // 2)
    # top, bottom = 0, delta_h
    # left, right = 0, delta_w
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = 0, delta_w

    color = 255
    try:
        color = image[0][0]
    except Exception as e:
        print(e)

    if color < 230:
        color = 230
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(color))
    crop_bb = (left / image.shape[1], top / image.shape[0], (image.shape[1] - right - left) / image.shape[1],
               (image.shape[0] - bottom - top) / image.shape[0])
    image[image > 230] = 255
    return image, crop_bb


def crop_handwriting_page(image, bb, image_size):
    '''
    Given an image and bounding box (bb) crop the input image based on the bounding box.
    The final output image was scaled based on the image size.

    Parameters
    ----------
    image: np.array
        Input form image

    bb: (x, y, w, h)
        The bounding box in percentages to crop

    image_size: (h, w)
        Image size to scale the output image to.

    Returns
    -------
    output_image: np.array
        cropped image of size image_size.
    '''
    image = crop_image(image, bb)

    image, _ = resize_image(image, desired_size=image_size)
    return image

def loadImage(img_file):
    img = io.imread(img_file)  # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:, :, :3]
    img = np.array(img)

    return img


def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)

    target_h, target_w = target_h32, target_w32

    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    return proc

# def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
#     height, width, channel = img.shape
#
#     # magnify image size
#     target_size = mag_ratio * max(height, width)
#
#     # set original image size
#     if target_size > square_size:
#         target_size = square_size
#
#     ratio = target_size / max(height, width)
#
#     target_h, target_w = int(height * ratio), int(width * ratio)
#     proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
#
#     # make canvas and paste image
#     target_h32, target_w32 = target_h, target_w
#     if target_h % 32 != 0:
#         target_h32 = target_h + (32 - target_h % 32)
#     if target_w % 32 != 0:
#         target_w32 = target_w + (32 - target_w % 32)
#     resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
#     resized[0:target_h, 0:target_w, :] = proc
#     target_h, target_w = target_h32, target_w32
#
#     size_heatmap = (int(target_w / 2), int(target_h / 2))
#
#     return resized, ratio, size_heatmap


def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def base64_to_cv2(base64_str, rgb=False):
    """
    base64 to numpy
    :param base64_str:
    :return:
    """
    imgString = base64.b64decode(base64_str)
    nparr = np.frombuffer(imgString, np.uint8)
    # BGR
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def bytes_to_cv2(img_bytes, rgb=False):
    """
    bytes to numpy
    :param img_bytes:
    :return:
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    # BGR
    imageOrg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if rgb:
        image = cv2.cvtColor(imageOrg, cv2.COLOR_BGR2RGB)
    return image,imageOrg
