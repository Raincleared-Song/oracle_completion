import cv2
import torch
import numpy as np
from utils import pad_image
from config import ConfigSharpen as Config


def process_sharpen(batch, mode):
    noise_batch, label_batch = [], []
    for noise_path, label_path in batch:
        noise_img = process_sharpen_single(noise_path)
        if mode != 'test':
            label_img = process_sharpen_single(label_path)
            assert noise_img.shape == label_img.shape == Config.pad_shape
        else:
            label_img = label_path
            assert noise_img.shape == Config.pad_shape
        noise_batch.append(noise_img)
        label_batch.append(label_img)
    return {
        'inputs': torch.FloatTensor(noise_batch),
        'labels': torch.FloatTensor(label_batch) if mode != 'test' else label_batch
    }


def process_sharpen_single(img_path):
    """img_path -> (channel_num, height, width)"""
    img = cv2.imread(img_path, -1)
    assert img.ndim in (2, 3)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)
    else:
        assert img.shape[2] <= 4
        img = np.transpose(img, (2, 0, 1))
    img = pad_image(img, Config.pad_shape)
    return img
