#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Fit Plane in pointcloud

Author: FILL IN
MatrNr: FILL IN
"""

from typing import List, Tuple, Callable

import copy

import numpy as np
import open3d as o3d


def fit_plane(pcd: o3d.geometry.PointCloud,
              confidence: float,
              inlier_threshold: float,
              min_sample_distance: float,
              error_func: Callable) -> Tuple[np.ndarray, np.ndarray, int]:
    """ Find dominant plane in pointcloud with sample consensus.

    Detect a plane in the input pointcloud using sample consensus. The number of iterations is chosen adaptively.
    The concrete error function is given as an parameter.

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. distance of a point from the plane to be considered an inlier (in meters)
    :type inlier_threshold: float

    :param min_sample_distance: Minimum distance of all sampled points to each other (in meters). For robustness.
    :type min_sample_distance: float

    :param error_func: Function to use when computing inlier error defined in error_funcs.py
    :type error_func: Callable

    :return: (best_plane, best_inliers, num_iterations)
        best_plane: Array with the coefficients of the plane equation ax+by+cz+d=0 (shape = (4,))
        best_inliers: Boolean array with the same length as pcd.points. Is True if the point at the index is an inlier
        num_iterations: The number of iterations that were needed for the sample consensus
    :rtype: tuple (np.ndarray[a,b,c,d], np.ndarray, int)
    """
    ######################################################
    # Write your own code here

    points = np.asarray(pcd.points)
    best_plane = np.array([0., 0., 1., 0.])
    best_inliers = np.full(points.shape[0], False)
    num_iterations = 0

    return best_plane, best_inliers, num_iterations


def filter_planes(pcd: o3d.geometry.PointCloud,
                  min_points_prop: float,
                  confidence: float,
                  inlier_threshold: float,
                  min_sample_distance: float,
                  error_func: Callable) -> Tuple[List[np.ndarray],
                                                 List[o3d.geometry.PointCloud],
                                                 o3d.geometry.PointCloud]:
    """ Find multiple planes in the input pointcloud and filter them out.

    Find multiple planes by applying the detect_plane function multiple times. If a plane is found in the pointcloud,
    the inliers of this pointcloud are filtered out and another plane is detected in the remaining pointcloud.
    Stops if a plane is found with a number of inliers < min_points_prop * number of input points.

    :param pcd: The (down-sampled) pointcloud in which to detect planes
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param min_points_prop: The proportion of points of the input pointcloud which have to be inliers of a plane for it
        to qualify as a valid plane.
    :type min_points_prop: float

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers for each plane.
    :type confidence: float

    :param inlier_threshold: Max. distance of a point from a plane to be considered an inlier (in meters).
    :type inlier_threshold: float

    :param min_sample_distance: Minimum distance of all sampled points to each other (in meters). For robustness.
    :type min_sample_distance: float

    :param error_func: Function to use when computing inlier error defined in error_funcs.py
    :type error_func: Callable

    :return: (plane_eqs, plane_pcds, filtered_pcd)
        plane_eqs is a list of np.arrays each holding the coefficient of a plane equation for one of the planes
        plane_pcd is a list of pointclouds with each holding the inliers of one plane
        filtered_pcd is the remaining pointcloud of all points which are not part of any of the planes
    :rtype: (List[np.ndarray], List[o3d.geometry.PointCloud], o3d.geometry.PointCloud)
    """
    ######################################################
    # Write your own code here
    plane_eqs = [np.array([0., 0., 1., 0.])]
    plane_pcds = []
    filtered_pcd = copy.deepcopy(pcd)

    return plane_eqs, plane_pcds, filtered_pcd
