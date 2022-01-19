#! python3
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 直方图均衡增强
def hist(image):
    r, g, b = cv2.split(image)
    r1 = cv2.equalizeHist(r)
    g1 = cv2.equalizeHist(g)
    b1 = cv2.equalizeHist(b)
    image_equal_clo = cv2.merge([r1, g1, b1])
    return image_equal_clo


# 拉普拉斯算子
def laplacian(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image_lap = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return image_lap


# 对数变换
def log(image):
    image_log = np.uint8(np.log(np.array(image) + 1))
    cv2.normalize(image_log, image_log, 0, 255, cv2.NORM_MINMAX)
    # 转换成8bit图像显示
    cv2.convertScaleAbs(image_log, image_log)
    return image_log


# 伽马变换
def gamma(image):
    fgamma = 2
    image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)
    return image_gamma


# 限制对比度自适应直方图均衡化CLAHE
def clahe(image):
    b, g, r = cv2.split(image)
    clahe_res = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe_res.apply(b)
    g = clahe_res.apply(g)
    r = clahe_res.apply(r)
    image_clahe = cv2.merge([b, g, r])
    return image_clahe


def replace_zeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


# retinex SSR
def ssr(src_img, size):
    l_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replace_zeroes(src_img)
    l_blur = replace_zeroes(l_blur)

    dst_img = cv2.log(img / 255.0)
    dst_lblur = cv2.log(l_blur / 255.0)
    dst_ixl = cv2.multiply(dst_img, dst_lblur)
    log_r = cv2.subtract(dst_img, dst_ixl)

    dst_r = cv2.normalize(log_r, None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_r)
    return log_uint8


def ssr_image(image):
    size = 3
    b_gray, g_gray, r_gray = cv2.split(image)
    b_gray = ssr(b_gray, size)
    g_gray = ssr(g_gray, size)
    r_gray = ssr(r_gray, size)
    result = cv2.merge([b_gray, g_gray, r_gray])
    return result


# retinex MMR
def msr(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_r = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replace_zeroes(img)
        l_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        l_blur = replace_zeroes(l_blur)
        dst_img = cv2.log(img / 255.0)
        dst_lblur = cv2.log(l_blur / 255.0)
        dst_ixl = cv2.multiply(dst_img, dst_lblur)
        log_r += weight * cv2.subtract(dst_img, dst_ixl)

    dst_r = cv2.normalize(log_r, None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_r)
    return log_uint8


def msr_image(image):
    scales = [15, 101, 301]  # [3,5,9]
    b_gray, g_gray, r_gray = cv2.split(image)
    b_gray = msr(b_gray, scales)
    g_gray = msr(g_gray, scales)
    r_gray = msr(r_gray, scales)
    result = cv2.merge([b_gray, g_gray, r_gray])
    return result


def main():
    i = 1
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    file_ll = range(1, 6)
    for file_id in file_ll:
        ll = len(file_ll)
        filename = str(file_id) + '.jpg'

        image = cv2.imread("./images" + "/" + str(filename), -1)
        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        plt.subplot(ll, 3, i)
        plt.imshow(image)
        plt.axis('off')
        plt.title('img_raw')

        # retinex_ssr
        i = i + 1
        image_ssr = ssr_image(image)
        image_ssr = cv2.bilateralFilter(image_ssr, 0, 100, 5)
        image_ssr = cv2.filter2D(image_ssr, cv2.CV_8UC3, kernel)
        image_ssr = cv2.cvtColor(image_ssr, cv2.COLOR_BGR2BGRA)
        _, image_ssr = cv2.threshold(image_ssr, 150, 255, cv2.THRESH_BINARY)
        image_ssr = cv2.GaussianBlur(image_ssr, (13, 13), 0)
        plt.subplot(ll, 3, i)
        plt.imshow(image_ssr)
        plt.axis('off')
        plt.title('SSR')
        i = i + 1

        # retinex_msr
        image_msr = msr_image(image)
        image_msr = cv2.bilateralFilter(image_msr, 0, 100, 5)
        image_msr = cv2.filter2D(image_msr, cv2.CV_8UC3, kernel)
        image_msr = cv2.cvtColor(image_msr, cv2.COLOR_BGR2BGRA)
        _, image_msr = cv2.threshold(image_msr, 130, 255, cv2.THRESH_BINARY)
        image_msr = cv2.GaussianBlur(image_msr, (13, 13), 0)
        plt.subplot(ll, 3, i)
        plt.imshow(image_msr)
        plt.axis('off')
        plt.title('MSR')

        i = i + 1

    plt.savefig('images/demo.jpg')


if __name__ == '__main__':
    data = cv2.imread('images/demo.jpg', -1)
    print(data.shape)
    exit()
    main()
