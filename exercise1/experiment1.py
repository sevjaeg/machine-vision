#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Experiment 1 for the documentation

Author: Severin Jäger
MatrNr: 01613004
"""
from pathlib import Path
import cv2
import numpy as np
from blur_gauss import blur_gauss
from helper_functions import *
from hyst_thresh import hyst_thresh
from non_max import non_max
from sobel import sobel

save_image = True
matplotlib_plotting = False

current_path = Path(__file__).parent
img_gray = cv2.imread(str(current_path.joinpath("image/rubens.jpg")), cv2.IMREAD_GRAYSCALE)
img_gray = img_gray.astype(np.float32) / 255.

blurred = []

# 1.1
for sigma in range(1, 4):
    for kernel_size in range(3, 25, 6):
        img_blur = blur_gauss(img_gray, sigma, kernel_width=kernel_size)
        show_image(img_blur, "Blurred Image s={:d} k={:d}".format(sigma, kernel_size), save_image=save_image, use_matplotlib=matplotlib_plotting)
    blurred.append(blur_gauss(img_gray, sigma))
    show_image(blurred[sigma-1], "Blurred Image s={:d}".format(sigma), save_image=save_image, use_matplotlib=matplotlib_plotting)


# 1.2
plot_row_intensities(img_gray, 400)
plot_row_intensities(blurred[0], 400)
plot_row_intensities(blurred[1], 400)
plot_row_intensities(blurred[2], 400)

plot_row_intensities(img_gray, 600)
plot_row_intensities(blurred[0], 600)
plot_row_intensities(blurred[1], 600)
plot_row_intensities(blurred[2], 600)

# 1.3
gradients, orientations = sobel(img_gray)
edges = non_max(gradients, orientations)
show_image(edges, "Edges Original Image", save_image=save_image, use_matplotlib=matplotlib_plotting)

gradients, orientations = sobel(blurred[0])
edges_blurred1 = non_max(gradients, orientations)
show_image(edges_blurred1, "Edges Blurred Image s=1", save_image=save_image, use_matplotlib=matplotlib_plotting)
gradients, orientations = sobel(blurred[1])
edges_blurred2 = non_max(gradients, orientations)
show_image(edges_blurred2, "Edges Blurred Image s=2", save_image=save_image, use_matplotlib=matplotlib_plotting)
gradients, orientations = sobel(blurred[2])
edges_blurred3 = non_max(gradients, orientations)
show_image(edges_blurred3, "Edges Blurred Image s=3", save_image=save_image, use_matplotlib=matplotlib_plotting)

# 1.4
canny_edges = hyst_thresh(edges, 0.25, 0.5)
show_image(canny_edges, "Canny Edges Original", save_image=save_image, use_matplotlib=matplotlib_plotting)
canny_edges = hyst_thresh(edges_blurred1, 0.25, 0.5)
show_image(canny_edges, "Canny Edges Blurred s=1", save_image=save_image, use_matplotlib=matplotlib_plotting)
canny_edges = hyst_thresh(edges_blurred2, 0.25, 0.5)
show_image(canny_edges, "Canny Edges Blurred s=2", save_image=save_image, use_matplotlib=matplotlib_plotting)
canny_edges = hyst_thresh(edges_blurred3, 0.25, 0.5)
show_image(canny_edges, "Canny Edges Blurred s=3", save_image=save_image, use_matplotlib=matplotlib_plotting)

cv2.destroyAllWindows()
