#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Blur the input image with Gaussian filter kernel

Author: Severin Jäger
MatrNr: 01613004
"""

import cv2
import numpy as np


def blur_gauss(img: np.array, sigma: float) -> np.array:
    """ Blur the input image with a Gaussian filter with standard deviation of sigma.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma: The standard deviation of the Gaussian kernel
    :type sigma: float

    :return: Blurred image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0.,1.]
    """
    ######################################################

    kernel_width = 2 * round(3 * sigma) + 1

    kernel = np.zeros(shape=(kernel_width, kernel_width))
    for i, row in enumerate(kernel):
        x = i - np.floor(kernel_width / 2)
        for j, cell in enumerate(row):
            y = j - np.floor(kernel_width / 2)
            kernel[i, j] = 1 / (2 * np.pi * np.power(sigma, 2)) * \
                np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2)))

    print("Applying Gauß filter with kernel")
    print(kernel)

    # TODO exact normalization?
    print(np.sum(kernel.flatten()))

    # TODO border type
    img_blur = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    ######################################################
    return img_blur
