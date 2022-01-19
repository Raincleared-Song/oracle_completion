import os
import cv2
import numpy as np
from cv2 import dnn_superres
from skimage.util import random_noise
from utils import *


def delete_jut(img, u_threshold: int, v_threshold: int, jut_black: bool):
    """
    消除图像突出部
    :param img: 输入图像
    :param u_threshold: 突出部宽度阈值
    :param v_threshold: 突出部高度阈值
    :param jut_black: 突出部颜色是否是黑色
    :return:
    """
    if jut_black:
        jc, oc = 0, 255
    else:
        jc, oc = 255, 0
    height, width = img.shape
    for i in range(height - 1):
        ptr = img[i, :]
        for j in range(width - 1):
            # 行消除
            if ptr[j] == oc and ptr[j + 1] == jc:
                if j + u_threshold >= width:
                    ptr[(j + 1): width] = oc
            else:
                k = j + 2
                for k in range(j + 2, j + u_threshold + 1):
                    if ptr[k] == oc:
                        break
                if k == width or ptr[k] == oc:
                    ptr[(j + 1): k] = oc
    for j in range(width - 1):
        ptr = img[:, j]
        for i in range(height - 1):
            # 列消除
            if ptr[i] == oc and ptr[i + 1] == jc:
                if i + v_threshold >= height:
                    ptr[(i + 1): height] = oc
            else:
                k = i + 2
                for k in range(i + 2, i + v_threshold + 1):
                    if ptr[k] == oc:
                        break
                if k == height or ptr[k] == oc:
                    ptr[(i + 1): k] = oc


def image_binary(path: str):
    """二值化、去噪、超分辨"""
    # 灰度图
    img = cv2.imread(path, 0)
    base, fmt = os.path.splitext(path)
    # 自适应二值化
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 1)
    img = cv2.medianBlur(img, 5)  # 中值滤波

    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # _, img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # _, img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # _, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    # img = 255 - img
    # erode_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (0, 0))
    # img = cv2.erode(img, erode_element)
    # img = 255 - img
    # delete_jut(img, 1, 1, True)

    # 转三原色图
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # 超分辨率
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel('pretrain_models/ESPCN_x4.pb')
    sr.setModel('espcn', 4)
    img = sr.upsample(img)
    # 转灰度图，第二次二值化
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)

    cv2.imwrite(base + '_cv' + fmt, img)


def image_add_noise(path: str):
    """降灰度、高斯平滑、高斯噪声"""
    # 灰度图
    img = cv2.imread(path, -1)
    base, fmt = os.path.splitext(path)
    # 降灰度 (<130 -> 100)
    mask = img < 130
    img += (mask * 80).astype(np.uint8)
    # 高斯平滑
    img = cv2.GaussianBlur(img, (49, 49), 0)
    # 加噪声
    img = random_noise(img, mode='gaussian', seed=100)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(base + '_noise' + fmt, img)


if __name__ == '__main__':
    img1 = cv2.imread('oracle/01005/2/01005-3592-2-0_cv.png', 0)
    img2 = cv2.imread('oracle/01003/2/01003-3589-2-6_cv.png', 0)
    img3 = cv2.imread('oracle/01005/2/01005-3592-2-1_cv.png', 0)
    img4 = cv2.imread('oracle/01003/2/01003-3589-2-11_cv.png', 0)
    img5 = cv2.imread('oracle/01001/1/01001-3587-1-2_cv.png', 0)
    # 梯度直方图 + 余弦距离效果最好
    print(similarity_hog(img1, img2, (500, 500)))
    print(similarity_average_hash(img1, img2))
    print(similarity_difference_hash(img1, img2))
    print(similarity_perceptual_hash(img1, img2))
    print(similarity_hog(img1, img5, (500, 500)))
    print(similarity_average_hash(img1, img5))
    print(similarity_difference_hash(img1, img5))
    print(similarity_perceptual_hash(img1, img5))

    print(similarity_hog(img3, img4, (500, 500)))
    print(similarity_average_hash(img3, img4))
    print(similarity_difference_hash(img3, img4))
    print(similarity_perceptual_hash(img3, img4))
    print(similarity_hog(img3, img5, (500, 500)))
    print(similarity_average_hash(img3, img5))
    print(similarity_difference_hash(img3, img5))
    print(similarity_perceptual_hash(img3, img5))
    exit()
    image_binary('oracle/01001/1/01001-3587-1-2.png')
    image_binary('oracle/01005/2/01005-3592-2-0.png')
    image_binary('oracle/01003/2/01003-3589-2-6.png')
    image_binary('oracle/01005/2/01005-3592-2-1.png')
    image_binary('oracle/01003/2/01003-3589-2-11.png')
    # image_binary('oracle/01010/1/01010-3604-1-1.png')
    # image_add_noise('oracle/01010/1/01010-3604-1-1_cv.png')
