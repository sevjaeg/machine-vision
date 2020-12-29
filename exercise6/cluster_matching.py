# -*- coding: utf-8 -*-

""" Functions to convert point clouds to images

Author: Severin Jäger
MatrNr: 01613004
"""

from typing import List, Tuple

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2


def map_matches_to_cluster(matches, keypoints, cluster, max_label, image_clusters, cmap):
    ret = 0
    for m in matches:
        match = m[0]
        kp = keypoints[match.trainIdx]
        position = tuple(np.round(kp.pt).astype(np.int))
        is_valid = False
        for u in [-1, 0, 1]:
            for v in [-1, 0, 1]:
                x, y = position
                x = x + u
                y = y + v
                is_valid = is_in_cluster((x, y), cluster, max_label, image_clusters, cmap) or is_valid
        if is_valid:
            # TODO weights
            # print(position, cluster)
            ret += 1

    return ret


def is_in_cluster(position, cluster, max_label, image_clusters, cmap):
    colour = np.multiply(cmap(cluster/(max_label if max_label > 0 else 1)), 255).astype(np.uint8)[:3][..., ::-1]
    y, x = position
    pixel = image_clusters[x, y]
    #print(position)
    #print(pixel)
    return np.all(pixel == colour)


def is_border_pixel(position, image):
    return False


def get_cluster_coordinates(image_clusters, max_label, cmap):
    size = max_label + 1
    labels = np.array(range(size))
    centers = np.zeros((size, 2))
    for i in range(size):
        colour = np.multiply(cmap(labels[i] / (max_label if max_label > 0 else 1)), 255).astype(np.uint8)[:3][..., ::-1]
        pos = np.argwhere(image_clusters == colour)
        centers[i] = np.average(pos[:,(0,1)], axis=0)
    return centers


