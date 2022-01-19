import cv2
import numpy as np
from cv2 import dnn_superres
from skimage.util import random_noise


def pad_image(img, shape: tuple, pad_color: int = 255):
    """给图片做 padding 以使他们大小一致"""
    img_shape, ori_shape = img.shape, shape
    assert len(img_shape) == len(shape)
    # 若有 channel 维度，则是第 0 维
    if len(shape) == 3:
        assert img_shape[0] == shape[0]
        shape, img_shape = shape[1:], img_shape[1:]
    new_h, new_w = shape
    h, w = img_shape
    assert h <= new_h and w <= new_w
    pad_h, pad_w = (new_h - h) // 2, (new_w - w) // 2
    new_img = np.full(ori_shape, pad_color)
    if len(ori_shape) == 3:
        new_img[:, pad_h:pad_h+h, pad_w:pad_w+w] = img
    else:
        new_img[pad_h:pad_h + h, pad_w:pad_w + w] = img
    assert new_img.shape == ori_shape
    return new_img


sr_model = None


def binary_sr(img):
    """对图像进行二值化、超分辨率操作"""
    # 自适应二值化
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 1)
    img = cv2.medianBlur(img, 5)  # 中值滤波
    # 转三原色图
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # 超分辨率
    global sr_model
    if sr_model is None:
        print('loading super resolution model ......')
        sr_model = dnn_superres.DnnSuperResImpl_create()
        sr_model.readModel('pretrain_models/ESPCN_x4.pb')
        sr_model.setModel('espcn', 4)
    img = sr_model.upsample(img)
    # 转灰度图，第二次二值化
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    return img


def add_noise(img):
    """对图像降灰度、高斯平滑、加高斯噪声"""
    # 降灰度 (<130 -> 100)
    mask = img < 130
    img = img + (mask * 80).astype(np.uint8)
    # 高斯平滑
    img = cv2.GaussianBlur(img, (49, 49), 0)
    # 加噪声
    img = random_noise(img, mode='gaussian', seed=100)
    img = (img * 255).astype(np.uint8)
    return img
