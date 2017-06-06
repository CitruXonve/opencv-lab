#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from time import clock


def main():
    print('Specify input image:', end='')

    try:
        filename = raw_input()
    except:
        filename = input()

    origin = cv2.imread(filename, cv2.IMREAD_COLOR)
    origin = cv2.GaussianBlur(origin, (0, 0), 0.8)
    # origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

    # cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
    # cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
    # cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel



    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(origin, (x, y), 2, 255, -1)

    plt.imshow(origin)
    plt.show()

if __name__ == '__main__':
    main()
