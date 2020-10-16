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

    # Sectors (gradient direction)
    # A: 0° to 45° and -180° to -135°
    # B: 45° to 90° and -135° to -90°
    # C: 90° to 135° and -90° to -45°
    # D: 135° to 180° and -45° to 0°
    in_sector_a = np.logical_or(np.logical_and(orientations >= 0, orientations < np.pi/4),
                                orientations <= -3*np.pi/4)
    in_sector_b = np.logical_or(np.logical_and(orientations >= np.pi/4, orientations < np.pi/2),
                                np.logical_and(orientations >= -3*np.pi/4, orientations < -np.pi/2))
    in_sector_c = np.logical_or(np.logical_and(orientations >= np.pi/2, orientations < 3*np.pi/4),
                                np.logical_and(orientations >= -np.pi/2, orientations < -np.pi/4))
    in_sector_d = np.logical_or(orientations >= 3*np.pi/4,
                                np.logical_and(orientations >= -np.pi/4, orientations < 0))

    d_a = gradients.copy()
    d11_a = np.roll(d_a, axis=(0, 1), shift=(-1, -1))  # D(x+1,y+1)
    d12_a = np.roll(d_a, axis=0, shift=-1)  # D(x+1,y)
    d1_a = n_y * d11_a + (n_x - n_y) * d12_a  # not divided by n_x yet
    d21_a = np.roll(d_a, axis=(0, 1), shift=(1, 1))  # D(x-1,y-1)
    d22_a = np.roll(d_a, axis=0, shift=1)  # D(x-1,y)
    d2_a = n_y * d21_a + (n_x - n_y) * d22_a  # not divided by n_x yet
    d_a = n_x*d_a  # avoid division by n_x (potentially 0)
    is_max_a = np.logical_and(d_a > d1_a, d_a > d2_a)
    edges_a = np.where(np.logical_and(in_sector_a, is_max_a), gradients,
                       np.zeros(shape=gradients.shape).astype(np.float32))

    d_b = gradients.copy()
    d11_b = np.roll(d_b, axis=0, shift=-1)  # D (x+1,y+1)
    d12_b = np.roll(d_b, axis=1, shift=-1)  # D(x,y+1)
    d1_b = n_x*d11_b + (n_y-n_x)*d12_b  # not divided by n_y yet
    d21_b = np.roll(d_b, axis=(0, 1), shift=(1, 1))  # D(x-1,y-1)
    d22_b = np.roll(d_b, axis=1, shift=1)  # D(x,y-1)
    d2_b = n_x*d21_b + (n_y-n_x)*d22_b   # not divided by n_y yet
    d_b = n_y * d_b  # avoid division by n_y (potentially 0)
    is_max_b = np.logical_and(d_b > d1_b, d_b > d2_b)
    edges_b = np.where(np.logical_and(in_sector_b, is_max_b), gradients,
                       np.zeros(shape=gradients.shape).astype(np.float32))

    d_c = gradients.copy()
    d11_c = np.roll(d_c, axis=(0, 1), shift=(1, -1))  # D (x-1,y+1)
    d12_c = np.roll(d_c, axis=1, shift=-1)  # D(x,y+1)
    d1_c = n_x * d11_c + (n_y - n_x) * d12_c  # not divided by n_y yet
    d21_c = np.roll(d_c, axis=(0, 1), shift=(-1, +1))  # D(x+1,y-1)
    d22_c = np.roll(d_c, axis=1, shift=1)  # D(x,y-1)
    d2_c = n_x * d21_c + (n_y - n_x) * d22_c  # not divided by n_y yet
    d_c = n_y * d_c  # avoid division by n_y (potentially 0)
    is_max_c = np.logical_and(d_c > d1_c, d_c > d2_c)
    edges_c = np.where(np.logical_and(in_sector_c, is_max_c), gradients,
                       np.zeros(shape=gradients.shape).astype(np.float32))

    d_d = gradients.copy()
    d11_d = np.roll(d_d, axis=(0, 1), shift=(1, -1))  # D (x-1,y+1)
    d12_d = np.roll(d_d, axis=0, shift=1)  # D(x-1,y)
    d1_d = n_y * d11_d + (n_x - n_y) * d12_d  # not divided by n_x yet
    d21_d = np.roll(d_d, axis=(0, 1), shift=(-1, 1))  # D(x+1,y-1)
    d22_d = np.roll(d_d, axis=0, shift=-1)  # D(x+1,y)
    d2_d = n_y * d21_d + (n_x - n_y) * d22_d  # not divided by n_x yet
    d_d = n_x * d_d  # avoid division by n_x (potentially 0)
    is_max_d = np.logical_and(d_d > d1_d, d_d > d2_d)
    edges_d = np.where(np.logical_and(in_sector_d, is_max_d), gradients,
                       np.zeros(shape=gradients.shape).astype(np.float32))

    edges = edges_a + edges_b + edges_c + edges_d
    ######################################################

    return edges
