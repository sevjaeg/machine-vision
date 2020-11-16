#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Detect an object in an image and return position, scale and orientation

Author: Severin Jäger
MatrNr: 01613004
"""
from typing import Tuple, List

import numpy as np
import cv2

import sklearn.cluster
import matplotlib.pyplot as plt

from helper_functions import *


def detect_objects(scene_img: np.ndarray,
                   object_img: np.ndarray,
                   scene_keypoints: List[cv2.KeyPoint],
                   object_keypoints: List[cv2.KeyPoint],
                   matches: List[List[cv2.DMatch]],
                   debug_output: bool = False) -> np.ndarray:
    """Return detected configurations of object_img in scene_img given keypoints and matches

    In this function you should implement the whole object detection pipeline. First, filter out bad matches.
    Then extract the position, scale, and orientation of all object hypotheses. Store them in a voting space.
    Cluster the data points of the voting space using one of the clustering algorithms of sklearn.cluster.
    You will need to filter the clusters further to arrive at concrete object hypotheses.
    The object recognition does not have to work perfectly for all provided images,
    but you should be able to explain in the documentation why some errors occur.

    :param scene_img: The color image where the object should be detected.
    :type scene_img: np.ndarray with shape (height, width, 3) with dtype = np.uint8 and values in the range [0, 255]

    :param object_img: An image of the object to be detected
    :type object_img: np.ndarray with shape (height, width, 3) with dtype = np.uint8 and values in the range [0, 255]

    :param scene_keypoints: List of all detected SIFT keypoints in the scene_img
    :type scene_keypoints: list of cv2.KeyPoint https://docs.opencv.org/trunk/d2/d29/classcv_1_1KeyPoint.html

    :param object_keypoints: List of all detected SIFT keypoints in the object_img
    :type object_keypoints: list of cv2.KeyPoint https://docs.opencv.org/trunk/d2/d29/classcv_1_1KeyPoint.html

    :param matches: Holds all the possible matches. Each 'row' are matches of one object_keypoint to scene_keypoints
    :type matches: list of lists of cv2.DMatch https://docs.opencv.org/master/d4/de0/classcv_1_1DMatch.html

    :param debug_output: If True enables additional output of intermediate steps.
        [You can use this parameter for your own debugging efforts, but you don't have to use it for the submission.]
    :type debug_output: bool

    :return: An array with shape (n, 4) with each row holding one detected object configuration (x, y, s, o)
        x, y: coordinates of the top-left corner of the detected object in the scene image coordinate frame
        s: relative scale of the object between object coordinate frame and scene image coordinate frame
        o: orientation (clockwise)
    :rtype: np.array with shape (n, 4) with n being the number of detected objects
    """
    ######################################################
    # Parameters
    SIFT_THRESHOLD = 350
    EPS = 0.08
    MIN_SAMPLES = 11
    MIN_CLUSTER_DIST = 20

    voting_space = np.zeros((0, 4))
    for matches_obj in matches:
        for match in matches_obj:
            if match.distance > SIFT_THRESHOLD:                            # Filter out bad SIFT matches
                continue
            obj_kp = object_keypoints[match.queryIdx]
            scn_kp = scene_keypoints[match.trainIdx]
            vote = np.asarray(match_to_params(scn_kp, obj_kp))  # calculate vote for this match
            voting_space = np.r_[voting_space, [vote]]          # add vote to voting space
    print("Voting space with {:d} elements".format(voting_space.shape[0]))

    # Clustering
    clustering_space = np.zeros((voting_space.shape[0], 4))  # Normalised voting space
    clustering_space[:, 0] = voting_space[:, 0] / max(voting_space[:, 0])
    clustering_space[:, 1] = voting_space[:, 1] / max(voting_space[:, 1])
    clustering_space[:, 2] = voting_space[:, 2] / max(voting_space[:, 2])
    clustering_space[:, 3] = voting_space[:, 3] / max(voting_space[:, 3])

    # 0.08, 11 for all images
    cluster_labels = sklearn.cluster.DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(clustering_space)
    number_of_clusters = np.max(cluster_labels).astype(int) + 1
    print("{:d} clusters detected".format(number_of_clusters))

    object_configurations = np.zeros((0, 4))
    for label in range(number_of_clusters):
        cluster = voting_space[cluster_labels == label, :]
        # median of all cluster members is kept
        object_configurations = np.r_[object_configurations, [np.median(cluster, axis=0)]]
        if debug_output:  # show clusters
            plot_img = draw_rectangles(scene_img, object_img, cluster)
            show_image(plot_img, "cluster")

    for cluster_a in range(0, number_of_clusters):
        for cluster_b in range(cluster_a + 1, number_of_clusters):              # iterate over all cluster combinations
            # find clusters close to each other and average their values
            if np.linalg.norm(object_configurations[cluster_a, :] - object_configurations[cluster_b, :]) \
                    < MIN_CLUSTER_DIST:
                object_configurations[cluster_a, [0, 2]] = np.average([object_configurations[cluster_a, [0, 2]],
                                                                       object_configurations[cluster_b, [0, 2]]],
                                                                      axis=0)
                object_configurations[cluster_a, 3] = average_angles(object_configurations[cluster_a, 3],
                                                                     object_configurations[cluster_b, 3])
                object_configurations[cluster_b, :] = 0                          # set lines to be removed to zero
    object_configurations = object_configurations[object_configurations.all(1)]  # get rid of all zero lines
    ######################################################

    return object_configurations


def match_to_params(scene_keypoint: cv2.KeyPoint, object_keypoint: cv2.KeyPoint) -> Tuple[float, float, float, float]:
    """ Compute the position, rotation and scale of an object implied by a matching pair of descriptors

    This function uses two matching keypoints in the object and scene image to calculate the x and y coordinates, the
    scale and the orientation of the object in the scene image. The scale factor determines the relative size of the
    object image in the scene image.
    The orientation is the rotation of the object in the scene image in clockwise direction in rad.
    A rotation of 0 would be in x-direction and means no relative orientation change between object and scene image.

    :param scene_keypoint: Keypoint in the scene_img where we want to detect the object
    :type scene_keypoint: cv2.KeyPoint https://docs.opencv.org/trunk/d2/d29/classcv_1_1KeyPoint.html

    :param object_keypoint: Keypoint in the object_img
    :type object_keypoint: cv2.KeyPoint https://docs.opencv.org/trunk/d2/d29/classcv_1_1KeyPoint.html

    :return: (x, y, scale, orientation)
        x,y: coordinates of top-left corner of detected object in the scene image
        s: relative scale
        o: orientation (clockwise)
    :rtype: (float, float, float, float)
    """
    ######################################################
    x_scene, y_scene = scene_keypoint.pt
    x_object, y_object = object_keypoint.pt

    r = np.sqrt(np.square(x_object) + np.square(y_object))
    beta = np.arctan2(y_object, x_object) + np.pi

    scale = scene_keypoint.size / object_keypoint.size
    orientation = np.pi/180 * (scene_keypoint.angle - object_keypoint.angle)
    # keep the angle between -pi and pi
    if orientation < -np.pi:
        orientation += 2*np.pi
    elif orientation > np.pi:
        orientation -= 2*np.pi

    x = x_scene + scale * r * np.cos(beta + orientation)
    y = y_scene + scale * r * np.sin(beta + orientation)
    ######################################################
    return x, y, scale, orientation


def average_angles(a, b):
    """
    Calculates the average of two angles (input and output in radians [-pi, pi])
    """
    x = np.cos(a) + np.cos(b)
    y = np.sin(a) + np.sin(b)
    return np.arctan2(y, x)
