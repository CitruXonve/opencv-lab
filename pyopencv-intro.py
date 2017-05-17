# coding:utf-8
import numpy as np
import cv2
import matplotlib.pyplot as plt


def my_inv(src):
    width, height, ch = src.shape
    res = src.copy()
    # res=np.zeros((height,width,ch), np.uint8)
    # print height,width,ch
    for i in range(width):
        for j in range(height):
            # print(src[i,j])
            res[i, j] = (255 - src[i, j][0], 255 -
                         src[i, j][1], 255 - src[i, j][2])
    return res


def my_merge(src, append):
    res = src.copy()
    width, height, ch = src.shape
    for i in range(width):
        for j in range(height):
            for c in range(ch):
                base = src[i, j][c]
                blend = append[i, j][c]
                # Photoshop混合模式之颜色减淡
                if blend < 255:
                    # warning: ... byte - overflow
                    tmp = base + (base * int(blend)) / (255 - blend)
                    res[i, j][c] = min(tmp, 255)
                else:
                    res[i, j][c] = base
    return res


def main():
    # filename = 'img/papers.co-ar10-cute-girl-illustration-anime-sky-33-iphone6-wallpaper.jpg'
    filename = 'img/20170509205642.jpg'

    origin = cv2.imread(filename, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(origin)
    origin = cv2.merge([r, g, b])

    # cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
    # cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
    # cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel

    inv = origin.copy()
    inv = cv2.bitwise_not(inv)
    # inv=my_inv(inv)
    blur = cv2.GaussianBlur(inv, (5, 5), 0)

    plt.figure(num='anime')

    plt.subplot(1, 3, 1)
    plt.title('origin image')
    plt.imshow(origin)
    plt.axis('off')  # 不显示坐标尺寸

    plt.subplot(1, 3, 2)
    plt.title('inv && GaussianBlur')
    plt.imshow(blur)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('merge')
    plt.imshow(my_merge(origin, blur))
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()
