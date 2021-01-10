# -*- coding: utf-8 -*-

""" Helper functions for displaying images and point clouds

Author: Severin Jäger
MatrNr: 01613004
"""

from typing import List

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2


def show_image(img: np.array, title: str, save_image: bool = False, use_matplotlib: bool = False) -> None:
    """ Plot an image with either OpenCV or Matplotlib. Adapted from Exercise 1

    :param img: :param img: Input image with shape (height, width) or (height, width, channels)
    :param title: The title of the plot which is also used as a filename if save_image is chosen
    :param save_image: If this is set to True, an image will be saved to disc as title.png
    :param use_matplotlib: If this is set to True, Matplotlib will be used for plotting, OpenCV otherwise
    """

    # First check if img is color or grayscale. Raise an exception on a wrong type.
    if len(img.shape) == 3:
        is_color = True
    elif len(img.shape) == 2:
        is_color = False
    else:
        raise ValueError(
            'The image does not have a valid shape. Expected either (height, width) or (height, width, channels)')

    if use_matplotlib:
        plt.figure()
        plt.title(title)
        if is_color:
            # OpenCV uses BGR order while Matplotlib uses RGB. Reverse the the channels to plot the correct colors
            plt.imshow(img[..., ::-1])
        else:
            plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        cv2.imshow(title, img)
        cv2.waitKey(0)

    if save_image:
        cv2.imwrite(title.replace(" ", "_") + ".png", img)


def show_pcds(plane_pcds: List[o3d.geometry.PointCloud]) -> None:
    """
    Plot multiple point clouds in different colours. Adapted from Exercise 5
    :param plane_pcds: The list of point clouds to be displayed
    """
    # Color the individual point clouds in different colors
    colormap = plt.cm.get_cmap("gist_rainbow", len(plane_pcds))
    for i, plane_pcd in enumerate(plane_pcds):
        plane_pcd.paint_uniform_color(colormap(i)[0:3])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for p in plane_pcds:
        vis.add_geometry(p)
    vc = vis.get_view_control()
    vc.set_front([-0.0, 0.0, -1])
    vc.set_lookat([0, -0.0, 1])
    vc.set_up([0, -1, 0])
    vc.set_zoom(0.5)
    vis.run()
    vis.destroy_window()
