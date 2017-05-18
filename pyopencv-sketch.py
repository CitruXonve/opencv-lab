#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import clock


def my_inv(src):
    width, height, ch = src.shape
    res = src.copy()
    for i in range(width):
        for j in range(height):
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
                overlay = append[i, j][c]
                # Photoshop混合模式之颜色减淡
                if overlay < 255:
                    # tmp = base + (base * int(overlay)) / (255 - overlay)
                    res[i, j][c] = min((base << 8) // (255 - overlay), 255)
                else:
                    res[i, j][c] = 255
    return res


def my_merge2(src, append):
    res = src.copy()
    width, height, ch = src.shape
    for i in range(width):
        for j in range(height):
            for c in range(ch):
                base = src[i, j][c]
                overlay = append[i, j][c]
                # Photoshop混合模式之变亮
                res[i, j][c] = max(base, overlay)
    return res


def createRGBColorTable(Highlight, Midtones, Shadow, OutputHighlight, OutputShadow):
    diff = int(Highlight - Shadow)
    outDiff = int(OutputHighlight - OutputShadow)

    if not((Highlight <= 255 and diff <= 255 and diff >= 2) or
            (OutputShadow <= 255 and OutputHighlight <= 255 and outDiff < 255) or
            (not(Midtones > 9.99 and Midtones > 0.1) and Midtones != 1.0)):
        raise Exception('Invalid level parameters!')

    coef = 255.0 / diff
    outCoef = outDiff / 255.0
    exponent = 1.0 / Midtones

    colorTable = [0] * 256

    for i in range(0, 256):
        colorTable[i] = i
        # calculate black field and white field of input level
        if colorTable[i] <= Shadow:
            v = 0
        else:
            v = int((colorTable[i] - Shadow) * coef + 0.5)
            if v > 255:
                v = 255
        # calculate midtone field of input level
        v = int(pow(v / 255.0, exponent) * 255.0 + 0.5)
        # calculate output level
        colorTable[i] = int(v * outCoef + OutputShadow + 0.5)

    return colorTable


def adjustRGBLevel(src, Highlight, Midtones, Shadow, OutputHighlight, OutputShadow):
    width, height, channels = src.shape
    dst = np.zeros(src.shape, np.uint8)

    try:
        colorTable = createRGBColorTable(
            Highlight, Midtones, Shadow, OutputHighlight, OutputShadow)
    except:
        raise Exception('Invalid colorTable!')

    for i in range(width):
        for j in range(height):
            for c in range(channels):
                dst[i, j][c] = colorTable[src[i, j][c]]

    return dst


def get_file_name(filename):
    tmp = filename.split('.')
    return tmp[0]


def get_file_extension(filename):
    tmp = filename.split('.')
    return tmp[len(tmp) - 1]


def main():
    print('Specify input image:./img/', end='')
    img_dir = 'img/'
    data_dir = 'data/'
    try:
        filename = raw_input()
    except:
        filename = input()

    before_process = clock()

    origin = cv2.imread(img_dir + filename, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(origin)
    origin = cv2.merge([r, g, b])

    # cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
    # cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
    # cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel

    if get_file_extension(filename) != 'png':
        origin = cv2.bilateralFilter(origin, 5, 80, 120)

    inv = origin.copy()
    inv = cv2.bitwise_not(inv)
    # inv=my_inv(inv)
    blur = cv2.GaussianBlur(inv, (0, 0), 0.8)
    merge = my_merge(origin, blur)
    merge_r = my_merge(blur, origin)
    merge_2p = my_merge2(merge, merge_r)
    merge_2pa = adjustRGBLevel(merge_2p, 255, 0.8, 128, 255, 0)

    after_process = clock()

    plt.figure(num='anime' + ' time elapsed: %.3f s' %
               (after_process - before_process))

    plt.subplot(2, 4, 1)
    plt.title('Orinigal image')
    plt.imshow(origin)
    plt.axis('off')  # 不显示坐标尺寸

    plt.subplot(2, 4, 2)
    plt.title('Inverse && GaussianBlur')
    plt.imshow(blur)
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.title('Merged')
    plt.imshow(merge)
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.title('Merged-Reverse')
    plt.imshow(merge_r)
    plt.axis('off')

    plt.subplot(2, 4, 5)
    plt.title('Merged-2p')
    plt.imshow(merge_2p)
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.title('Merged-2p+')
    plt.imshow(merge_2pa)
    plt.axis('off')

    plt.show()

    r, g, b = cv2.split(merge)
    merge = cv2.merge([b, g, r])
    cv2.imwrite(data_dir + get_file_name(filename) +
                '.' + get_file_extension(filename), merge)

    r, g, b = cv2.split(merge_r)
    merge_r = cv2.merge([b, g, r])
    cv2.imwrite(data_dir + get_file_name(filename) +
                '-r.' + get_file_extension(filename), merge_r)

    r, g, b = cv2.split(merge_2p)
    merge_2p = cv2.merge([b, g, r])
    cv2.imwrite(data_dir + get_file_name(filename) +
                '-2p.' + get_file_extension(filename), merge_2p)

    r, g, b = cv2.split(merge_2pa)
    merge_2pa = cv2.merge([b, g, r])
    cv2.imwrite(data_dir + get_file_name(filename) +
                '-2p+.' + get_file_extension(filename), merge_2pa)


if __name__ == '__main__':
    main()
