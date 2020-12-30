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

from camera_params import *
from util import *

def calculate_pixel_positions(pcd):
    """
    Calculate the pixel positions of each point in the point cloud
    """
    points = np.asarray(pcd.points)
    points_homogeneous = np.c_[points, np.ones((points.shape[0], 1))]
    pixels = np.matmul(K, points_homogeneous.T)

    # transform to normal (non-homogeneous coordinates)
    pixels /= np.where(pixels[2, :] != 0.0, pixels[2, :], 1)

    # there might be negative pixel values, ensure all values are at least 0
    pixels[0, :] += np.max(-pixels[0, :])
    pixels[1, :] += np.max(-pixels[1, :])

    return pixels

def create_image_from_pcd(pcd, downsampling=1, show=False):
    """
    Convert a point cloud to an image
    """
    bgr_scene = np.multiply(np.asarray(pcd.colors), 255).astype(np.uint8)[..., ::-1]
    # Calculate pixel for each point
    pixels = calculate_pixel_positions(pcd)

    # reduce the image size. This is not used any longer
    pixels[(0, 1), :] /= downsampling
    pixels = np.round(pixels).astype(np.int)

    # create and fill the image
    image_size = (np.max(pixels[1]) - np.min(pixels[1]) + 1, np.max(pixels[0]) - np.min(pixels[0]) + 1, 3)
    image = np.zeros(image_size, np.uint8)
    image[pixels[1, :], pixels[0, :], :] = bgr_scene

    if show:
        print("created an " + str(image_size) + " image")
        show_image(image, "Image from pcd", save_image=False)

    return image