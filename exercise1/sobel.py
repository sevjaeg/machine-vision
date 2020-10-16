#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Edge detection with the Sobel filter

Author: Severin Jäger
MatrNr: 01613004
"""

import cv2
import numpy as np


def sobel(img: np.array) -> (np.array, np.array):
    """ Apply the Sobel filter to the input image and return the gradient and the orientation.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    :return: (gradient, orientation): gradient: edge strength of the image in range [0.,1.],
                                      orientation: angle of gradient in range [-np.pi, np.pi]
    :rtype: (np.array, np.array)
    """
    ######################################################
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.transpose(kernel_x)

    print("Applying Sobel filter")

    g_x = cv2.filter2D(img, -1, kernel_x)
    g_y = cv2.filter2D(img, -1, kernel_y)

    gradient = np.sqrt(np.power(g_x, 2) + np.power(g_y, 2))
    orientation = np.arctan2(g_x, g_y)  # not g_y/g_x as in the slides to stay consistent with x and y
    ######################################################
    return gradient, orientation
