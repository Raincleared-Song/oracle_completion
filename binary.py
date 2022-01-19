#! python3
# -*- coding:utf-8 -*-
import cv2


def binary(img_dir):
    name, fmt = img_dir.split(".")
    img = cv2.imread(img_dir, 0)
    # img = cv2.medianBlur(img, 5)  # 中值滤波
    # 自适应阈值二值化
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 1)
    img = cv2.medianBlur(img, 5)  # 中值滤波
    # cv2.imshow("img", img)
    cv2.imwrite(name + "_cv." + fmt, img)
    print(img_dir + " process successfully!")


def main():
    binary("images/binary1.png")


if __name__ == '__main__':
    main()
