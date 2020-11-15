#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Detect an object in an image and return position, scale and orientation

Author: FILL IN
MatrNr: FILL IN
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
                   matches: List[cv2.DMatch],
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
    # Write your own code here

    # Replace these lines.
    object_configurations = np.empty((0, 4))  # Accomplishes the same as np.zeros((1,4)), for demonstration of np.r_
    object_configurations = np.r_[object_configurations, np.array([[0., 0., 0., 0.]])]

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
    # Write your own code here
    x, y, scale, orientation = 0., 0., 0., 0.  # Replace this line

    ######################################################
    return x, y, scale, orientation
