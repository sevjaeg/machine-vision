# -*- coding: utf-8 -*-

""" Helper functions

Author: Severin Jäger
MatrNr: 01613004
"""

from typing import List, Tuple

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def plot_pcds(plane_pcds: List[o3d.geometry.PointCloud]) -> None:
    colormap = plt.cm.get_cmap("gist_rainbow", len(plane_pcds))
    # Color the individual plane pointclouds in different colors
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

