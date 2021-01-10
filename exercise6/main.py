# -*- coding: utf-8 -*-

""" Open Challenge Main Function

Author: Severin Jäger
MatrNr: 01613004
"""

from pathlib import Path
import copy
import functools
import operator
import time
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

from util import *
from points_to_image import *
from cluster_matching import *

debug = False
input_image = "9"

categories = {
        0: 'unknown',
        1: 'book',
        2: 'cookiebox',
        3: 'cup',
        4: 'ketchup',
        5: 'sugar',
        6: 'sweets',
        7: 'tea'
    }

if __name__ == '__main__':
    start = time.time()

    # Load test point cloud
    current_path = Path(__file__).parent
    pcd = o3d.io.read_point_cloud(str(current_path.joinpath("test/image00" + input_image + ".pcd")))
    if not pcd.has_points():
        raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

    # 1. Remove ground plane
    # Parameters ############
    ransac_inlier_dist = 0.015
    ransac_iterations = 7000
    #########################
    ground_plane_model, ground_plane_inliers = pcd.segment_plane(distance_threshold=ransac_inlier_dist,
                                                                 ransac_n=3,
                                                                 num_iterations=ransac_iterations)
    ground_plane_pcd = pcd.select_by_index(ground_plane_inliers)
    ground_plane_pcd.paint_uniform_color([1.0, 0, 0])
    objects_pcd = pcd.select_by_index(ground_plane_inliers, invert=True)

    if debug:
        show_pcds([ground_plane_pcd, copy.deepcopy(objects_pcd)])  # pass copy as the colours are modified

    # 2. Project remaining points to 2D image
    # Parameters ############
    #########################
    image = create_image_from_pcd(objects_pcd, show=debug)

    # 3. Clustering of the 3D space
    # Parameters ############
    dbscan_eps = 0.02
    dbscan_min_points = 50
    cluster_min_points = 300
    voxel_size = 0.002
    #########################
    # Down-sample the object point cloud to reduce computation time
    pcd_sampled = objects_pcd.voxel_down_sample(voxel_size=voxel_size)

    labels = np.array(pcd_sampled.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=False))
    max_label = labels.max()
    no_clusters = max_label + 1
    print(f"{no_clusters} clusters detected")
    if debug:
        # color the labels in the point cloud (these colours are used as label encoding in the following)
        colour_map = plt.get_cmap("tab20")
        cluster_colours = colour_map(labels / (max_label if max_label > 0 else 1))
        cluster_colours[labels < 0] = 0
        pcd_sampled.colors = o3d.utility.Vector3dVector(cluster_colours[:, :3])
        o3d.visualization.draw_geometries([pcd_sampled])

    # remove small clusters (this yields better results than a high min_points value for the dbscan)
    left_clusters = []
    for cluster in range(max_label + 1):
        if np.count_nonzero(labels == cluster) < cluster_min_points:
            labels[np.argwhere(labels == cluster)] = -1
        else:
            left_clusters.append(cluster)

    # clean up labels (keep between 0 and the new max_label)
    labels_filtered = copy.deepcopy(labels)
    for i, left_cluster in enumerate(left_clusters):
        labels_filtered[np.argwhere(labels == left_cluster)] = i
    no_clusters = len(left_clusters)
    max_label = no_clusters - 1

    # color the labels in the point cloud (these colours are used as label encoding in the following)
    colour_map = plt.get_cmap("tab20")
    cluster_colours = colour_map(labels_filtered / (max_label if max_label > 0 else 1))[:, :3]
    cluster_colours[labels_filtered < 0] = 0
    pcd_sampled.colors = o3d.utility.Vector3dVector(cluster_colours)

    print(f"{no_clusters} clusters remaining")
    if debug:
        o3d.visualization.draw_geometries([pcd_sampled])

    # 4. Project clusters to 2D image
    # Parameters ############
    kernel_size = 7
    #########################
    image_labels = create_image_from_pcd(pcd_sampled, show=debug)
    # Fill the holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    image_labels_filled = cv2.morphologyEx(image_labels, cv2.MORPH_CLOSE, kernel)

    # calculate cluster centers for labels
    cluster_centers = get_cluster_coordinates(image_labels_filled, max_label, colour_map)
    if debug:
        image_labels_filled_debug = copy.deepcopy(image_labels_filled)
        for i in range(no_clusters):
            image_labels_filled_debug = cv2.putText(image_labels_filled_debug, str(i),
                                                    tuple(np.flip(cluster_centers[i, :]).astype(np.int)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        show_image(image_labels_filled_debug, "filled")

    # 5. Match features
    # Parameters ############
    sift_threshold = 400
    min_matches = 4
    considered_matches = 3
    neighbourhood_size = 3
    use_distance_weights = False
    use_colour_sift = True
    #########################
    # data structure holding the classification results of the cluster
    # for each cluster, there is a tuple (category, category rating)
    matching_results = no_clusters * [(0, 0.0)]

    sift = cv2.SIFT_create()
    # Interest points in the scene
    if use_colour_sift:
        scene_b = image[:, :, 0]
        scene_g = image[:, :, 1]
        scene_r = image[:, :, 2]
        scene_kp_b, scene_desc_b = sift.detectAndCompute(scene_b, None)
        scene_kp_g, scene_desc_g = sift.detectAndCompute(scene_g, None)
        scene_kp_r, scene_desc_r = sift.detectAndCompute(scene_r, None)
    else:
        scene_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scene_kp, scene_desc = sift.detectAndCompute(scene_gray, None)

    path_training = Path('training')
    paths = path_training.rglob('*.pcd')
    for path in paths:  # iterate over all files
        training_pcd = o3d.io.read_point_cloud(str(current_path.joinpath(path)))
        if not training_pcd.has_points():
            raise FileNotFoundError("Couldn't load point cloud in " + str(current_path.joinpath(path)))
        # convert the point cloud to an image
        training_image = create_image_from_pcd(training_pcd, show=False)

        # Interest points in the training image
        if use_colour_sift:
            training_b = training_image[:, :, 0]
            training_g = training_image[:, :, 1]
            training_r = training_image[:, :, 2]
            training_kp_b, training_desc_b = sift.detectAndCompute(training_b, None)
            training_kp_g, training_desc_g = sift.detectAndCompute(training_g, None)
            training_kp_r, training_desc_r = sift.detectAndCompute(training_r, None)
        else:
            training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)
            training_kp, training_desc = sift.detectAndCompute(training_gray, None)

        # Matching of interest points, only consider best match (from exercise 3)
        index_params = dict(algorithm=1, trees=1)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        if use_colour_sift:
            matches_b = flann.knnMatch(training_desc_b, scene_desc_b, k=considered_matches)
            matches_g = flann.knnMatch(training_desc_g, scene_desc_g, k=considered_matches)
            matches_r = flann.knnMatch(training_desc_r, scene_desc_r, k=considered_matches)
            # get rid of matches with a distance greater the threshold, flatten list of lists into list
            matches_filtered_b = list(filter(lambda x: x.distance <= sift_threshold,
                                           functools.reduce(operator.iconcat, matches_b, [])))
            matches_filtered_g = list(filter(lambda x: x.distance <= sift_threshold,
                                           functools.reduce(operator.iconcat, matches_g, [])))
            matches_filtered_r = list(filter(lambda x: x.distance <= sift_threshold,
                                           functools.reduce(operator.iconcat, matches_r, [])))
            total_matches = len(matches_filtered_b) + len(matches_filtered_g) + len(matches_filtered_r)
        else:
            matches = flann.knnMatch(training_desc, scene_desc, considered_matches)
            # get rid of matches with a distance greater the threshold, flatten list of lists into list
            matches_filtered = list(filter(lambda x: x.distance <= sift_threshold,
                                           functools.reduce(operator.iconcat, matches, [])))
            total_matches = len(matches_filtered)

        # get category from file name
        cat_str = ''.join(list(filter(lambda c: c.isalpha(), path.stem)))
        category = list(categories.keys())[list(categories.values()).index(cat_str)]

        for cluster in range(no_clusters):
            # calculate the matching metric (ration between matches and total matches)
            # between the training image and each cluster
            if use_colour_sift:
                no_matches_b, weights_b = map_matches_to_cluster(matches_filtered_b, scene_kp_b, cluster, max_label,
                                                             image_labels_filled,
                                                             colour_map, neighbourhood=neighbourhood_size)
                no_matches_g, weights_g = map_matches_to_cluster(matches_filtered_g, scene_kp_g, cluster, max_label,
                                                                 image_labels_filled,
                                                                 colour_map, neighbourhood=neighbourhood_size)
                no_matches_r, weights_r = map_matches_to_cluster(matches_filtered_r, scene_kp_r, cluster, max_label,
                                                                 image_labels_filled,
                                                                 colour_map, neighbourhood=neighbourhood_size)
                no_matches = no_matches_b + no_matches_g + no_matches_r
                weights = weights_b + weights_g + weights_r
            else:   
                no_matches, weights = map_matches_to_cluster(matches_filtered, scene_kp, cluster, max_label, image_labels_filled,
                                                colour_map, neighbourhood=neighbourhood_size)
            if use_distance_weights:
                ratio = weights/total_matches if total_matches > 0 else 0.0
            else:
                ratio = no_matches / total_matches if total_matches > 0 else 0.0
            if no_matches >= min_matches and ratio > matching_results[cluster][1]:  # keep the best match
                matching_results[cluster] = (category, ratio)

        if False:  # plot matches
            match_mask = np.ones(np.array(matches).shape, dtype=np.int)
            draw_params = dict(matchesMask=match_mask,
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            matches_img = cv2.drawMatchesKnn(training_image,
                                             training_kp,
                                             image,
                                             scene_kp,
                                             matches,
                                             None,
                                             **draw_params)
            show_image(matches_img, "Matches")

    # print result and create image annotations
    for cluster in range(no_clusters):
        cat, perc = matching_results[cluster]
        print(categories[cat], perc)
        image = cv2.putText(image, categories[cat], tuple(np.flip(cluster_centers[cluster, :]).astype(np.int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    end = time.time()
    span = end - start
    print("{:.2f} s".format(span))

    show_image(image, "rgb" + input_image, save_image=True)
