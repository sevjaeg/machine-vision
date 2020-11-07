#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Descriptor functions

Author: Severin Jäger
MatrNr: 01613004
"""
from typing import Callable

import numpy as np
import cv2

from helper_functions import circle_mask


def patch_basic(patch: np.ndarray) -> np.ndarray:
    """ Return the basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    # quadratic matrix to vector
    return np.reshape(patch, (1, patch.size))
    ######################################################


def patch_norm(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    desc = patch_basic(patch)
    desc = desc / np.max(desc)
    return desc
    ######################################################


def patch_sort(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized and sorted basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    return np.sort(patch_norm(patch))
    ######################################################


def patch_sort_circle(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized and sorted basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    # values outside the circle are set to 0, this is acceptable as the array is sorted afterwards
    patch = np.where(circle_mask(patch.shape[0]), patch, 0)
    return patch_sort(patch)
    ######################################################


def block_orientations(patch: np.ndarray) -> np.ndarray:
    """ Compute orientation-histogram based descriptor from a patch

    Orientation histograms from 16 4 x 4 blocks of the patch, concatenated in row major order (1 x 128).
    Each orientation histogram should consist of 8 bins in the range [-pi, pi], each bin being weighted by the sum of
    gradient magnitudes of pixel orientations assigned to that bin.

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (16, 16)

    :return: Orientation-histogram based Descriptor
    :rtype: np.ndarray with shape (1, 128)
    """
    ######################################################
    dx, dy = np.gradient(patch)
    grad = np.sqrt(np.square(dx) + np.square(dy))
    orientations = np.arctan2(dy, dx)

    hist = np.zeros((1, 0))
    for x in range(4):
        for y in range(4):
            #block = patch[4*x:4*x+4, 4*y:4*y+4]
            grad_block = grad[4*x:4*x+4, 4*y:4*y+4]
            orientations_block = orientations[4*y:4*y+4, 4*y:4*y+4]
            # 8 bins of the histogram
            bins = np.array([-np.pi, -3/4*np.pi, -1/2*np.pi, -1/4*np.pi, 0, 1/4*np.pi, 1/2*np.pi, 3/4*np.pi])
            orientations_binned = np.digitize(orientations_block, bins)
            h = np.zeros((1, 8))
            for i in range(bins.size):
                # histogram weight = gradient (achieved by multiplication)
                orientations_i = np.multiply(np.where(orientations_binned == i+1, 1, 0), grad_block)
                h[0, i] = np.sum(orientations_i)
            hist = np.hstack((hist, h))
    return hist

    ######################################################


def compute_descriptors(descriptor_func: Callable,
                        img: np.ndarray,
                        locations: np.ndarray,
                        patch_size: int) -> (np.ndarray, np.ndarray):
    """ Calculate the given descriptor using descriptor_func on patches of the image, centred on the locations provided

    :param descriptor_func: Descriptor to compute at each location
    :type descriptor_func: function

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param locations: Locations at which to compute the descriptors (n x 2)
    :type locations: np.ndarray with the shape (n x 2). First column is y (row), second is x (column).

    :param patch_size: Value defining the width and height of the patch around each location
        to pass to the descriptor function.
    :type patch_size: int

    :return: (interest_points, descriptors):
        interest_points: k x 2 array containing the image coordinates [y,x] of the corners.
            Locations too close to the image boundary to cut out the image patch should not be contained.
        descriptors: k x patch_size^2 matrix containing the patch descriptors.
            Each row vector stores the concatenated column vectors of the image patch around each corner.
            Corners too close to the image boundary to cut out the image patch should not be contained.
    :rtype: (np.ndarray, np.ndarray)
    """
    ######################################################
    x_max = img.shape[0]
    y_max = img.shape[1]

    interest_points = np.zeros((1, 2))
    width = patch_size*patch_size
    if descriptor_func == block_orientations:
        width = 128
    descriptors = np.zeros((1, width))
    for location in locations:  # iterate over interestp oints
        x = location[0]
        y = location[1]
        if patch_size % 2 == 0:  # handle 16x16 patches by shifting the patch half a pixel to the bottom right
            d1 = int((patch_size - 1) / 2)
            d2 = int((patch_size - 1) / 2) + 1
        else:
            d1 = int((patch_size - 1) / 2)
            d2 = d1
        # only calculate patch if all patch pixels are within the image
        if (x-d1) > 0 and (x+d2) < x_max and (y-d1) > 0 and (y+d2) < y_max:
            patch = img[x-d1:x+d2+1, y-d1:y+d2+1]
            descriptor = descriptor_func(patch)
            interest_points = np.vstack((interest_points, np.array([x, y])))  # add element
            descriptors = np.vstack((descriptors, descriptor))  # add element

    interest_points = interest_points[1:, :]  # remove the first element (created before the loop)
    descriptors = descriptors[1:, :]  # remove the first element (created before the loop)

    return interest_points, descriptors
    ######################################################

