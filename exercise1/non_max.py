#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Non-Maxima Suppression

Author: Severin Jäger
MatrNr: 01613004
"""

import cv2
import numpy as np


def non_max(gradients: np.array, orientations: np.array) -> np.array:
    """ Apply Non-Maxima Suppression and return an edge image.

    Filter out all the values of the gradients array which are not local maxima.
    The orientations are used to check for larger pixel values in the direction of orientation.

    :param gradients: Edge strength of the image in range [0.,1.]
    :type gradients: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param orientations: angle of gradient in range [-np.pi, np.pi]
    :type orientations: np.array with shape (height, width) with dtype = np.float32 and values in the range [-pi, pi]

    :return: Non-Maxima suppressed gradients
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    """
    ######################################################
    print("Performing non-maximum suppression")
    n_x = np.absolute(np.cos(orientations))
    n_y = np.absolute(np.sin(orientations))

    # TODO treat border
    # TODO clean up

    not_shifted = np.logical_or(np.logical_and(orientations >= 0, orientations < np.pi/2), orientations < - np.pi/2)

    d = gradients.copy()
    d11 = np.roll(d, axis=0, shift=-1)  #D (x+1,y+1)
    # d11[d11.shape[0]-1, :] = 0
    #d11[:, 0] = 0

    d12 = np.roll(d, axis=1, shift=-1)  # D(x,y+1)
    #d12[:, 0] = 0

    d1 = n_x*d11 + (n_y-n_x)*d12  # not divided by n_y yet

    d21 = np.roll(d, axis=0, shift=1)  # D(x-1,y-1)
    d21 = np.roll(d21, axis=1, shift=1)
    # d21[0, :] = 0
    #d21[:, d21.shape[1]-1] = 0

    d22 = np.roll(d, axis=1, shift=1)  # D(x,y-1)
    #d22[:, d22.shape[1]-1] = 0

    d2 = n_x*d21 + (n_y-n_x)*d22   # not divided by n_y yet

    d = n_y * d  # avoid division by n_y (potentially 0)
    is_max = np.logical_and(d >= d1, d >= d2)

    edges_std = np.where(np.logical_and(not_shifted, is_max), gradients,
                         np.zeros(shape=gradients.shape).astype(np.float32))

    # TODO fix coords
    d_shifted = gradients.copy()
    d11_shifted = np.roll(d_shifted, axis=1, shift=-1)  # D (x+1,y+1)
    d11_shifted = np.roll(d11_shifted, axis=0, shift=-1)
    # d11[d11.shape[0]-1, :] = 0
    # d11[:, 0] = 0

    d12_shifted = np.roll(d_shifted, axis=0, shift=-1)  # D(x,y+1)
    # d12[:, 0] = 0

    d1_shifted = n_x * d11_shifted + (n_y - n_x) * d12_shifted  # not divided by n_y yet

    d21_shifted = np.roll(d_shifted, axis=1, shift=1)  # D(x-1,y-1)
    d21_shifted = np.roll(d21_shifted, axis=0, shift=1)
    # d21[0, :] = 0
    # d21[:, d21.shape[1]-1] = 0

    d22_shifted = np.roll(d_shifted, axis=0, shift=1)  # D(x,y-1)
    # d22[:, d22.shape[1]-1] = 0

    d2_shifted = n_x * d21_shifted + (n_y - n_x) * d22_shifted  # not divided by n_y yet

    d_shifted = n_y * d_shifted  # avoid division by n_y (potentially 0)
    is_max_shifted = np.logical_and(d_shifted >= d1_shifted, d_shifted >= d2_shifted)

    edges_shifted = np.where(np.logical_and(np.logical_not(not_shifted), is_max_shifted), gradients,
                             np.zeros(shape=gradients.shape).astype(np.float32))

    edges = edges_std + edges_shifted
    ######################################################

    return edges
