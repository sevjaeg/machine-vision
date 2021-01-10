# -*- coding: utf-8 -*-

""" Functions for cluster handling

Author: Severin Jäger
MatrNr: 01613004
"""

import numpy as np


def map_matches_to_cluster(matches, keypoints, cluster, max_label, image_clusters, cmap, neighbourhood=3):
    """
    Calculates the number of matches connected to a given cluster within a given image

    :param matches: A list of matches
    :param keypoints: A list of key points
    :param cluster: The number of the cluster of interest
    :param max_label: The number of the last cluster (thus the largest label)
    :param image_clusters: An image containing the clusters (encoded using the cluster numbers and the colour map)
    :param cmap: The colour map related to the image
    :param neighbourhood: The size of the surrounding of a cluster in which a match is still considered an inlier in pixels.
    :return: The number of matches linked to key points within the given cluster and the distance weighted number as tuple
    """

    ret = 0
    m = 0
    for match in matches:
        kp = keypoints[match.trainIdx]

        x, y = tuple(np.round(kp.pt).astype(np.int))
        if is_in_cluster((x, y), cluster, max_label, image_clusters, cmap, neighbourhood):
            m += 1  # count inliers
            if match.distance > 0:
                ret += 1/(match.distance ** 2)  # use 1/distance^2 as weight
    return m, ret


def is_in_cluster(position, cluster, max_label, image_clusters, cmap, neighbourhood):
    """
    Checks whether a given pixel is part of a specific cluster in an image

    :param position:
    :param cluster:
    :param max_label:
    :param image_clusters:
    :param cmap:
    :param neighbourhood:
    :return:
    """

    d = int((neighbourhood - 1) / 2)
    colour = np.multiply(cmap(cluster/(max_label if max_label > 0 else 1)), 255).astype(np.uint8)[:3][..., ::-1]
    y, x = position
    if is_valid_neighbourhood(position, d, image_clusters):
        pixel = image_clusters[x-d:x+d+1, y-d:y+d+1]
    else:
        pixel = image_clusters[x, y]  # very simple solution for problems with key points close to the image border
    return np.any(np.all(pixel == colour, axis=2))


def is_valid_neighbourhood(position, d, image):
    """
    Checks whether all pixels of a given neighbourhood are inside the passed image

    :param position: center of the neighbourhood as tuple of pixels (x, y)
    :param d: size of the neighbourhood (as distance from the center in pixels)
    :param image: the respective image
    :return: True if all pixels of the neighbourhood lie inside the image, False otherwise
    """
    x, y = position
    x_max = image.shape[1]
    y_max = image.shape[0]
    return not (x-d < 0 or y-d < 0 or x+d >= x_max or y+d >= y_max)


def get_cluster_coordinates(image_clusters, max_label, cmap):
    """
    Calculates the center coordinates of all clusters in the given image

    :param image_clusters: An image containing the clusters (encoded using the cluster numbers and the colour map)
    :param max_label: The number of the last cluster (thus the largest label)
    :param cmap: The colour map related to the image
    :return: A numpy array of shape (max_label+1, 2) where each row states the center coordinates of one cluster (in pixels)
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

