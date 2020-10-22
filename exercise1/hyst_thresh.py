#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Hysteresis thresholding

Author: Severin Jäger
MatrNr: 01613004
"""

import cv2
import numpy as np


def hyst_thresh(edges_in: np.array, low: float, high: float) -> np.array:
    """ Apply hysteresis thresholding.

    Apply hysteresis thresholding to return the edges as a binary image. All connected pixels with value > low are
    considered a valid edge if at least one pixel has a value > high.

    :param edges_in: Edge strength of the image in range [0.,1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low: Value below all edges are filtered out
    :type low: float in range [0., 1.]

    :param high: Value which a connected element has to contain to not be filtered out
    :type high: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################
    print("Applying hysteresis with t1 = {:.2f} and t2 = {:.2f}".format(low, high))

    edges = edges_in/np.max(edges_in)  # normalize
    edges = np.where(edges >= low, edges, 0)  # get rid of elements below the lower threshold

    # returns the number of connected components and an image with the connected components numbered
    (N, connected_edges) = cv2.connectedComponents((edges*255).astype(np.uint8), connectivity=8)

    bitwise_img = np.zeros(edges.shape)
    for i in range(1, N):  # iterate over connected components
        image_i = np.where(connected_edges == i, edges, 0)  # select the pixels belonging to the component
        if (image_i >= high).any():
            bitwise_img += image_i

    bitwise_img = np.where(bitwise_img > 0, 1, 0).astype(np.float32)

    ######################################################
    return bitwise_img
