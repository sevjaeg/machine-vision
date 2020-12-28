# -*- coding: utf-8 -*-

""" Open Challenge Main

Author: Severin Jäger
MatrNr: 01613004
"""

from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

from camera_params import *
from util import  *

debug = True

if __name__ == '__main__':
    # Load test point cloud
    current_path = Path(__file__).parent
    pcd = o3d.io.read_point_cloud(str(current_path.joinpath("test/image000.pcd")))
    if not pcd.has_points():
        raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

    # Down-sample the loaded point cloud to reduce computation time
    # Parameters ############
    uniform_k = 12
    voxel_size = 0.01
    #########################

    # pcd_sampled = pcd.uniform_down_sample(uniform_k)
    pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 1. Remove ground plane
    # Parameters ############
    ransac_inlier_dist = 0.015
    ransac_iterations = 10000
    #########################
    ground_plane_model, ground_plane_inliers = pcd_sampled.segment_plane(distance_threshold=ransac_inlier_dist, ransac_n=3, num_iterations=ransac_iterations)

    ground_plane_pcd = pcd_sampled.select_by_index(ground_plane_inliers)
    ground_plane_pcd.paint_uniform_color([1.0,0,0])
    objects_pcd = pcd_sampled.select_by_index(ground_plane_inliers, invert=True)

    if debug:
        plot_pcds([ground_plane_pcd, objects_pcd])

    # 2. Clustering of the 3D space
    # Parameters ############
    dbscan_eps = 0.05
    dbscan_min_points = 25
    #########################
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            objects_pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=debug))

    max_label = labels.max()
    print(f"{max_label + 1} clusters detected")

    if debug:  # from open3d example
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        objects_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([objects_pcd])

    # 3. Project remaining points to 2D image
    # Parameters ############

    #########################

    points_scene = np.asarray(pcd_sampled.points)
    print(points_scene.shape)

    # Create rgb array (n, 3)
    rgb_scene_o3d = np.multiply(np.asarray(pcd_sampled.colors), 255).astype(np.uint8)

    # Calculate pixel for each point

    points_scene_homogeneous = np.c_[points_scene, np.ones((points_scene.shape[0], 1))]
    pixels = np.matmul(np.matmul(A_d, Rt), points_scene_homogeneous.T)

    x = pixels[0]/pixels[2]
    y = pixels[1] / pixels[2]

    dx = np.max(x) - np.min(x)
    dy = np.max(y) - np.min(y)

    ratio = dx/dy

    print(dx, dy)
    print(ratio)

    print(A_d)
    print(Rt)

    # Correct color space
    # rgb_scene = cv2.cvtColor(rgb_scene_o3d, code=cv2.COLOR_RGB2BGR)

    # 4. Project labels to 2D image
    # Parameters ############

    #########################

    # 5. Match features
    # Parameters ############

    #########################
