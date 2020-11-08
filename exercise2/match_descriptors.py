#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Match descriptors in two images

Author: Severin Jäger
MatrNr: 01613004
"""
import numpy as np
import cv2


def match_descriptors(descriptors_1: np.ndarray, descriptors_2: np.ndarray, best_only: bool) -> np.ndarray:
    """ Find matches for patch descriptors

    :param descriptors_1: Patch descriptors of first image
    :type descriptors_1: np.ndarray with shape (m, n) containing m descriptors of length n

    :param descriptors_2: Patch descriptor of second image
    :type descriptors_2: np.ndarray with shape (m, n) containing m descriptors of length n

    :param best_only: If True, only keep the best match for each descriptor
    :type best_only: Boolean

    :return: Array representing the successful matches. Each row contains the indices of the matches descriptors
    :rtype: np.ndarray with shape (k, 2) with k being the number of matches
    """
    ######################################################

    # I did not use best_only, as its meaning is not clearly specified for me (and its not used by the calling function)

    matches = np.zeros((0, 2))
    for i, desc1 in enumerate(descriptors_1):
        best_idx = 0
        dist1 = np.finfo(np.float32).max
        dist2 = np.finfo(np.float32).max
        for j, desc2 in enumerate(descriptors_2):
            dist = np.linalg.norm(desc1-desc2)
            if dist < dist1:
                dist2 = dist1
                dist1 = dist
                best_idx = j
            elif dist < dist2:
                dist2 = dist
        if dist1/dist2 < 0.8:
            matches = np.vstack((matches, np.array([i, best_idx])))
    print("Found {:d} matches".format(matches.shape[0]))
    return matches
    ######################################################
