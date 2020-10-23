#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Experiment 3 for the documentation

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


save_image = True
matplotlib_plotting = False

current_path = Path(__file__).parent
img_gray = cv2.imread(str(current_path.joinpath("image/rubens.jpg")), cv2.IMREAD_GRAYSCALE)
img_gray = img_gray.astype(np.float32) / 255.

sigma = 1
img_blur = blur_gauss(img_gray, sigma)
gradients, orientations = sobel(img_blur)
edges = non_max(gradients, orientations)

low = 0.25
high = 0.35
canny_edges = hyst_thresh(edges, low, high)
show_image(canny_edges, "No noise", save_image=save_image,
           use_matplotlib=matplotlib_plotting)

# 3.1
for s in [0.02, 0.05, 0.1, 0.2, 0.5]:
    img_noise = add_gaussian_noise(img_gray, sigma=s)
    show_image(img_noise, "Noise {:.2f}".format(s), save_image=save_image,
               use_matplotlib=matplotlib_plotting)
    img_blur = blur_gauss(img_noise, sigma)
    gradients, orientations = sobel(img_blur)
    edges = non_max(gradients, orientations)
    low = 0.25
    high = 0.35
    canny_edges = hyst_thresh(edges, low, high)
    show_image(canny_edges, "Edges Noise {:.2f}".format(s), save_image=save_image,
               use_matplotlib=matplotlib_plotting)


# 3.2
for sigma in [1, 2, 3, 4]:
    img_noise = add_gaussian_noise(img_gray, sigma=0.2)
    img_blur = blur_gauss(img_noise, sigma)
    gradients, orientations = sobel(img_blur)
    edges = non_max(gradients, orientations)
    low = 0.25
    high = 0.35
    canny_edges = hyst_thresh(edges, low, high)
    show_image(canny_edges, "Edges Noise 0.2 blurred with {:.2f}".format(sigma), save_image=save_image,
               use_matplotlib=matplotlib_plotting)

cv2.destroyAllWindows()
