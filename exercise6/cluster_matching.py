# -*- coding: utf-8 -*-

""" Functions for cluster handling

Author: Severin Jäger
MatrNr: 01613004
"""

import numpy as np


def map_matches_to_cluster(matches, keypoints, cluster, max_label, image_clusters, cmap, neighbourhood=3):
    """
    Calculates the number of matches connected to a given cluster within an image

    :param matches:
    :param keypoints:
    :param cluster:
    :param max_label:
    :param image_clusters:
    :param cmap:
    :param neighbourhood:
    :return:
    """

    ret = 0
    for match in matches:
        kp = keypoints[match.trainIdx]
        x, y = tuple(np.round(kp.pt).astype(np.int))
        is_valid = is_in_cluster((x, y), cluster, max_label, image_clusters, cmap, neighbourhood)
        if is_valid:
            ret += 1  # count inliers
    return ret


def is_in_cluster(position, cluster, max_label, image_clusters, cmap, neighbourhood):
    """
    Checks whether a given pixel is part of a specific cluster in an image
    :param position:
    :param cluster:
    :param max_label:
    :param image_clusters:
    :param cmap:
    :return:
    """

    #if not is_border_pixel((x, y), image_clusters):  # skip pixels outside the image
    d = int((neighbourhood - 1) / 2)
    colour = np.multiply(cmap(cluster/(max_label if max_label > 0 else 1)), 255).astype(np.uint8)[:3][..., ::-1]
    y, x = position
    pixel = image_clusters[x-d:x+d+1, y-d:y+d+1]
    return np.any(np.all(pixel == colour, axis=2))


def is_border_pixel(position, image):
    """
    Checks whether a given position is inside the passed image
    :param position:
    :param image:
    :return:
    """
    x, y = position
    x_max = image.shape[1]
    y_max = image.shape[0]
    return x < 0 or y < 0 or x >= x_max or y >= y_max


def get_cluster_coordinates(image_clusters, max_label, cmap):
    """
    Calculates the center coordinates of all clusters in the given image
    :param image_clusters:
    :param max_label:
    :param cmap:
    :return:
    """
    size = max_label + 1
    labels = np.array(range(size))
    centers = np.zeros((size, 2))
    for i in range(size):
        colour = np.multiply(cmap(labels[i] / (max_label if max_label > 0 else 1)), 255).astype(np.uint8)[:3][..., ::-1]
        indices = np.where(np.all(image_clusters == colour, axis=-1))
        pos = np.asarray([indices[0].T, indices[1].T]).T
        centers[i] = np.median(pos, axis=0).astype(np.int)
    return centers

