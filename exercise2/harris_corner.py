#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Corner detection with the Harris corner detector

Author: Severin Jäger
MatrNr: 01613004
"""
import numpy as np
import cv2


def harris_corner(img, sigma1, sigma2, k, threshold):
    """ Detect corners using the Harris corner detector

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma1: Sigma for the first Gaussian filtering
    :type sigma1: float

    :param sigma2: Sigma for the second Gaussian filtering
    :type sigma2: float

    :param k: Coefficient for harris formula
    :type k: float

    :param threshold: corner threshold
    :type threshold: float

    :return: (i_xx, i_yy, i_xy, g_xx, g_yy, g_xy, h_dense, h_nonmax, corners):
        i_xx: squared input image filtered with derivative of gaussian in x-direction
        i_yy: squared input image filtered with derivative of gaussian in y-direction
        i_xy: Multiplication of input image filtered with derivative of gaussian in x- and y-direction
        g_xx: i_xx filtered by larger gaussian
        g_yy: i_yy filtered by larger gaussian
        g_xy: i_xy filtered by larger gaussian
        h_dense: Result of harris calculation for every pixel. Array of same size as input image.
            Values normalized to 0-1
        h_nonmax: Binary mask of non-maxima suppression. Array of same size as input image.
            1 where values are NOT suppressed, 0 where they are.
        corners: n x 3 array containing all detected corners after thresholding and non-maxima suppression.
            Every row vector represents a corner with the elements [y, x, d]
            (d is the result of the harris calculation)
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    """

    ######################################################
    # Creating DoG
    kernel_width1 = 2 * round(3 * sigma1) + 1
    kernel1 = cv2.getGaussianKernel(kernel_width1, sigma1)
    gauss1 = np.outer(kernel1, kernel1.transpose())
    dog_x = np.gradient(gauss1, axis=1)
    dog_y = np.gradient(gauss1, axis=0)

    # Applying DoG
    i_x = cv2.filter2D(img, -1, dog_x, borderType=cv2.BORDER_REPLICATE)
    i_y = cv2.filter2D(img, -1, dog_y, borderType=cv2.BORDER_REPLICATE)
    i_xx = np.square(i_x)
    i_yy = np.square(i_y)
    i_xy = np.multiply(i_x, i_y)

    # Gaussian blurring
    kernel_width2 = 2 * round(3 * sigma2) + 1
    kernel2 = cv2.getGaussianKernel(kernel_width2, sigma2)
    gauss2 = np.outer(kernel2, kernel2.transpose())
    g_xx = cv2.filter2D(i_xx, -1, gauss2, borderType=cv2.BORDER_REPLICATE)
    g_yy = cv2.filter2D(i_yy, -1, gauss2, borderType=cv2.BORDER_REPLICATE)
    g_xy = cv2.filter2D(i_xy, -1, gauss2, borderType=cv2.BORDER_REPLICATE)

    # Calculating Harris features
    # R is defined as R = det(M) - k * trace(M)^2
    # For a 2x2 matrix M = [[a,b],[c,d]] the following relations hold:
    #   det(M) = a*d-b=c
    #   trace(M) = a+d
    # These simplifications yield the following line:
    r = np.multiply(g_xx, g_yy) - np.square(g_xy) - k * np.square(g_xx + g_yy)

    # Normalisation and thresholding
    r = r / np.max(r)
    r = np.where(r >= threshold, r, 0)

    # Non-maximum suppression
    r_non_max = non_max(r)

    # Collect corner points
    corners = np.argwhere(r_non_max == 1)
    # non_max returns binary values -> corners holds all coordinates of interest points
    corners_final = np.zeros((corners.shape[0], 3))
    corners_final[:, [0, 1]] = corners
    # additional column for corner strength (from non-binary matrix r)
    corners_final[:, 2] = r[corners_final[:, 0].astype(int), corners_final[:, 1].astype(int)]

    print("Found {:d} corners".format(corners_final.shape[0]))
    return i_xx, i_yy, i_xy, g_xx, g_yy, g_xy, r, r_non_max, corners_final
    ######################################################


def non_max(corners: np.array) -> np.array:
    """ Apply Non-Maxima Suppression and return an corner image.

    Filter out all the values of the corners array which are not local maxima.

    :param corners: Harris corner strength of the image in range [0.,1.]
    :type corners: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :return: Non-Maxima suppressed corners
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    """
    ######################################################
    corners_e = np.roll(corners, axis=0, shift=-1)              # corners(x+1,y)
    corners_ne = np.roll(corners, axis=(0, 1), shift=(-1, -1))  # corners(x+1,y+1)
    corners_n = np.roll(corners, axis=1, shift=-1)              # corners(x,y+1)
    corners_nw = np.roll(corners, axis=(0, 1), shift=(1, -1))   # corners(x-1,y+1)
    corners_w = np.roll(corners, axis=0, shift=1)               # corners(x-1,y)
    corners_sw = np.roll(corners, axis=(0, 1), shift=(1, 1))    # corners(x-1,y-1)
    corners_s = np.roll(corners, axis=1, shift=1)               # corners(x,y-1)
    corners_se = np.roll(corners, axis=(0, 1), shift=(-1, 1))   # corners(x+1,y-1)

    # 1 if greater than all neighbours, 0 otherwise
    # >= treat the case of two neighbouring pixels with the same strength (otherwise none of them is detected as corner)
    corners_non_max = np.where(np.logical_and(corners >= corners_e, np.logical_and(corners >= corners_ne,
                            np.logical_and(corners >= corners_n, np.logical_and(corners >= corners_nw,
                            np.logical_and(corners > corners_w, np.logical_and(corners > corners_sw,
                            np.logical_and(corners > corners_s, corners > corners_se))))))), 1, 0)
    ######################################################
    return corners_non_max
