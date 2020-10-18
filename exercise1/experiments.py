#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Experiments for the documentation

Author: Severin Jäger
MatrNr: 01613004
"""
from pathlib import Path

import cv2
import numpy as np

from blur_gauss import blur_gauss
from helper_functions import *
from hyst_auto import hyst_thresh_auto
from hyst_thresh import hyst_thresh
from non_max import non_max
from sobel import sobel


# Define behavior of the show_image function. You can change these variables if necessary
save_image = True
matplotlib_plotting = False


current_path = Path(__file__).parent
img_gray = cv2.imread(str(current_path.joinpath("image/rubens.jpg")), cv2.IMREAD_GRAYSCALE)
img_gray = img_gray.astype(np.float32) / 255.

#### Experiment 1 ####################

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

exit(0)

# 1. Blur Image
sigma = 3  # Change this value
img_blur = blur_gauss(img_gray, sigma)
show_image(img_blur, "Blurred Image", save_image=save_image, use_matplotlib=matplotlib_plotting)

# 2. Edge Detection
gradients, orientations = sobel(img_blur)
orientations_color = cv2.applyColorMap(np.uint8((orientations.copy() + np.pi) / (2 * np.pi) * 255),
                                       cv2.COLORMAP_RAINBOW)
orientations_color = orientations_color.astype(np.float32) / 255.
gradient_img = np.append(cv2.cvtColor(gradients, cv2.COLOR_GRAY2BGR), orientations_color, axis=1)
show_image(gradient_img, "Gradients", save_image=save_image, use_matplotlib=matplotlib_plotting)

# 3. Non-Maxima Suppression
edges = non_max(gradients, orientations)

# 4. Hysteresis Thresholding
hyst_method_auto = True
if hyst_method_auto:
    canny_edges = hyst_thresh_auto(edges, 0.25, 0.1)
else:
    canny_edges = hyst_thresh(edges, 0.3, 0.7)
show_image(canny_edges, "Canny Edges", save_image=save_image, use_matplotlib=matplotlib_plotting)

# Overlay the found edges in red over the original image
img_gray_overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
img_gray_overlay[canny_edges == 1.0] = (0., 0., 1.)
show_image(img_gray_overlay, "Overlay", save_image=save_image, use_matplotlib=matplotlib_plotting)

# Destroy all OpenCV windows in case we have any open
cv2.destroyAllWindows()
