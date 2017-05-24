#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from time import clock


def main():
    print('Specify input image:./img/', end='')
    img_dir = 'img/'
    data_dir = 'data/'
    try:
        filename = raw_input()
    except:
        filename = input()

    origin = cv2.imread(img_dir + filename, cv2.IMREAD_GRAYSCALE)

    # cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
    # cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
    # cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel

    hist = cv2.calcHist([origin], [0], None, [256], [0, 256])
    equ = cv2.equalizeHist(origin)
    hist_equ = cv2.calcHist([equ], [0], None, [256], [0, 256])

    plt.subplot(221), plt.imshow(origin, 'gray'), plt.axis('off')
    plt.subplot(222), plt.plot(hist)
    plt.subplot(223), plt.imshow(equ, 'gray'), plt.axis('off')
    plt.subplot(224), plt.plot(hist_equ)
    plt.xlim([0, 256])

    plt.show()

if __name__ == '__main__':
    main()
