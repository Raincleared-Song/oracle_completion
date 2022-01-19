#! python3
# -*- coding:utf-8 -*-
import cv2


def edge(img_dir):
    name = img_dir.split(".")[0]
    fmt = img_dir.split(".")[1]
    img = cv2.imread(img_dir, 0)
    img = cv2.medianBlur(img, 5)  # 中值滤波
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    img = cv2.medianBlur(img, 5)
    img = cv2.medianBlur(img, 5)  # 中值滤波
    img = cv2.medianBlur(img, 5)  # 中值滤波
    # cv2.imshow("img", img)
    mi = []
    mx = []
    for col, i in enumerate(img):
        mi.append(100000)
        mx.append(-100000)
        for index, j in enumerate(i):
            if j == 0:
                mi[-1] = min(mi[-1], index)
                mx[-1] = max(mx[-1], index)
        for index in range(mi[-1] + 1, mx[-1]):
            img[col][index] = 255
    cv2.imwrite(name + "_cv." + fmt, img)
    print(img_dir + " process successfully!")


def main():
    edge("images/edge1.jpg")


if __name__ == '__main__':
    main()
