import cv2
import numpy as np
from skimage.feature import hog
from .normalize import pad_image


def average_hash(img, shape: tuple = (8, 8)):
    """输入灰度图，计算均值哈希相似度"""
    assert len(img.shape) == len(shape) == 2
    img = cv2.resize(img, shape)
    s = np.sum(img)
    hash_str = []
    # 求平均灰度
    avg = s / 64
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img[i, j] > avg:
                hash_str.append(1)
            else:
                hash_str.append(0)
    return np.array(hash_str)


def difference_hash(img, shape: tuple = (8, 8)):
    """输入灰度图，计算差值哈希相似度"""
    assert len(img.shape) == len(shape) == 2
    img = cv2.resize(img, (shape[0]+1, shape[1]))
    hash_str = []
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img[i, j] > img[i, j + 1]:
                hash_str.append(1)
            else:
                hash_str.append(0)
    return np.array(hash_str)


def perceptual_hash(img, shape: tuple = (32, 32), dct_shape: tuple = (8, 8)):
    """输入灰度图，计算感知哈希相似度"""
    assert len(img.shape) == len(shape) == 2
    img = cv2.resize(img, shape)
    img = cv2.dct(np.float32(img))
    img = img[0:dct_shape[0], 0:dct_shape[1]]

    hash_str = []
    avg = np.mean(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > avg:
                hash_str.append(1)
            else:
                hash_str.append(0)
    return np.array(hash_str)


def cmp_hash(hash1: np.ndarray, hash2: np.ndarray):
    """计算两个 hash 的相似度"""
    assert hash1.shape == hash2.shape and hash1.ndim == 1 and hash1.shape[0] > 0
    return np.sum(hash1 == hash2) / hash1.shape[0]


def similarity_average_hash(img1, img2):
    return cmp_hash(average_hash(img1), average_hash(img2))


def similarity_difference_hash(img1, img2):
    return cmp_hash(difference_hash(img1), difference_hash(img2))


def similarity_perceptual_hash(img1, img2):
    return cmp_hash(perceptual_hash(img1), perceptual_hash(img2))


def similarity_hog(img1, img2, pad_shape: tuple):
    img1 = pad_image(img1, pad_shape)
    img2 = pad_image(img2, pad_shape)
    hog1 = hog(img1)
    hog2 = hog(img2)
    assert hog1.shape == hog2.shape
    return np.dot(hog1, hog2) / (np.linalg.norm(hog1) * (np.linalg.norm(hog2)))
