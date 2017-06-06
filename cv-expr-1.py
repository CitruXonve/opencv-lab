#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from time import clock


def my_calcHist_grayscale(src, min, max):
    dst = np.zeros(max - min, np.uint32)
    width, height = src.shape
    for i in range(width):
        for j in range(height):
            dst[src[i, j]] += 1
    return dst


def my_equalizeHist_grayscale(src):
    width, height = src.shape
    hist = my_calcHist_grayscale(src, 0, 256)
    cumulate = np.zeros(256, np.uint32)
    for i in range(0, 256):
        for j in range(0, i + 1):
            cumulate[i] += hist[j]

    g = np.zeros(256, np.uint8)
    dst = np.zeros(src.shape, np.uint8)
    for i in range(0, 256):
        g[i] = int(256.0 * cumulate[i] / width / height + 0.5)

    for i in range(width):
        for j in range(height):
            dst[i, j] = g[src[i, j]]
    return dst


def main():
    print('Specify an input image:', end='')
    try:
        filename = raw_input()
    except:
        filename = input()

    origin = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
    # cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
    # cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel

    # hist = my_calcHist_grayscale(origin, 0, 256)
    hist = cv2.calcHist([origin], [0], None, [256], [0, 256])
    # equ = my_equalizeHist_grayscale(origin)
    equ = cv2.equalizeHist(origin)
    # hist_equ = my_calcHist_grayscale(equ, 0, 256)
    hist_equ = cv2.calcHist([equ], [0], None, [256], [0, 256])

    plt.subplot(221), plt.imshow(origin, 'gray'), plt.axis('off'), plt.title('Original image')
    plt.subplot(222), plt.plot(hist), plt.title('Hist')
    plt.subplot(223), plt.imshow(equ, 'gray'), plt.axis('off'), plt.title('Hist-equalized image')
    plt.subplot(224), plt.plot(hist_equ), plt.title('Equalized hist')
    plt.xlim([0, 256])

    plt.show()

if __name__ == '__main__':
    main()
