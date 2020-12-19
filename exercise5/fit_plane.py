#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Fit Plane in pointcloud

Author: Severin Jäger
MatrNr: 01613004
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
    points = np.asarray(pcd.points)
    number_of_points = points.shape[0]
    num_iterations = 0
    eps = 3/number_of_points  # the three selected points are inliers
    min_error = np.inf
    best_plane = np.array([0., 1., 0., 0.])
    best_inliers = np.full(number_of_points, False)
    correct_prob = 1
    while correct_prob >= (1-confidence):
        p1 = points[np.random.randint(0, number_of_points), :]
        p2 = points[np.random.randint(0, number_of_points), :]
        p3 = points[np.random.randint(0, number_of_points), :]

        if (np.linalg.norm(p1-p2) < min_sample_distance
                or np.linalg.norm(p1-p3) < min_sample_distance
                or np.linalg.norm(p2-p3) < min_sample_distance):
            continue

        # calculate plane
        # parameter representation: x = p1  + s(p2-p1) + t(p3-p1) = p1 + s vec1 + t vec2
        vec1 = p2 - p1
        vec2 = p3 - p1
        # normal vector (normalised)
        n = np.cross(vec1, vec2)
        n = n / np.linalg.norm(n)
        d = np.dot(n, p1)  # distance between plane and origin (as n is normalised)
        plane = np.array([n[0], n[1], n[2], -d])

        # calculate distances
        # this is possible as the plane is given in the Hesse normal form
        distances = abs(np.dot(points, n).T - d)

        # find inliers using the error function
        [error, inliers] = error_func(pcd, distances, inlier_threshold)
        no_inliers = np.sum(inliers)

        # in case a better plane was found keep it
        if error < min_error:
            best_plane = plane
            best_inliers = inliers
            eps = no_inliers/number_of_points

        num_iterations = num_iterations + 1
        correct_prob = (1 - eps ** 3) ** num_iterations
        # print("Max dist", np.max(distances))
        # print("Inliers", no_inliers)
        # print("Error", error)

    # Refine plane
    inlier_points = points[best_inliers]
    b = np.ones((inlier_points.shape[0], 1))
    lstsq = np.linalg.lstsq(inlier_points, b, rcond=None)
    best_plane = np.concatenate((lstsq[0].flatten(), [-1.0]))

    if best_plane[2] <= 0:  # surface normal pointing downwards -> invert vector
        # print("Inverting vector")
        best_plane = best_plane * -1

    # print("Iterations:", num_iterations)
    # print("Plane:", best_plane)
    ######################################################
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

    plane_eqs, plane_pcds = [], []
    filtered_pcd = copy.deepcopy(pcd)
    no_points = np.asarray(pcd.points).shape[0]

    while True:
        best_plane, best_inliers, _ = fit_plane(filtered_pcd, confidence, inlier_threshold,
                                                min_sample_distance, error_func)
        if sum(best_inliers) < min_points_prop * no_points:
            print(str(len(plane_eqs)) + " planes detected")
            return plane_eqs, plane_pcds, filtered_pcd
        list_remainder = np.argwhere(np.invert(best_inliers)).flatten().tolist()
        list_plane = np.argwhere(best_inliers).flatten().tolist()
        plane_pcd = filtered_pcd.select_by_index(list_plane)
        filtered_pcd = filtered_pcd.select_by_index(list_remainder)
        plane_eqs.append(best_plane)
        plane_pcds.append(plane_pcd)

