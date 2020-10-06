#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Blur the input image with Gaussian filter kernel

Author: Severin Jäger
MatrNr: 01613004
"""

import cv2
import numpy as np
from helper_functions import convolve2d


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
    # TODO improver performance
    kernel_1d = cv2.getGaussianKernel(3, sigma)
    kernel = np.matmul(kernel_1d, np.transpose(kernel_1d))
    print("Applying Gauß filter with kernel")
    print(kernel)
    #img_blur = convolve2d(img, kernel)

    img_blur = cv2.filter2D(img, -1, kernel)

    # TODO compare performance to reference gauß implementation
    # img_blur = cv2.GaussianBlur(img, (3, 3), sigma)

    ######################################################
    return img_blur
