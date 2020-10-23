#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Experiment 2 for the documentation

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
img_gray = cv2.imread(str(current_path.joinpath("image/beardman.jpg")), cv2.IMREAD_GRAYSCALE)
img_gray = img_gray.astype(np.float32) / 255.

img_gray2 = cv2.imread(str(current_path.joinpath("image/rubens.jpg")), cv2.IMREAD_GRAYSCALE)
img_gray2 = img_gray2.astype(np.float32) / 255.

img_gray3 = cv2.imread(str(current_path.joinpath("image/parliament.jpg")), cv2.IMREAD_GRAYSCALE)
img_gray3 = img_gray3.astype(np.float32) / 255.

sigma = 3
img_blur = blur_gauss(img_gray, sigma)
img_blur2 = blur_gauss(img_gray2, sigma)
img_blur3 = blur_gauss(img_gray3, sigma)

gradients, orientations = sobel(img_blur)
gradients2, orientations2 = sobel(img_blur2)
gradients3, orientations3 = sobel(img_blur3)

edges = non_max(gradients, orientations)
edges2 = non_max(gradients2, orientations2)
edges3 = non_max(gradients3, orientations3)

# 2.1 & 2.2
low = 0.1
high = 0.1
for i in range(4):
    for j in range(5-i):
        canny_edges = hyst_thresh(edges, low, high)
        canny_edges2 = hyst_thresh(edges2, low, high)
        canny_edges3 = hyst_thresh(edges3, low, high)
        show_image(canny_edges, "CannyA {:.2f} {:.2f}".format(low, high), save_image=save_image,
                   use_matplotlib=matplotlib_plotting)
        show_image(canny_edges2, "CannyB {:.2f} {:.2f}".format(low, high), save_image=save_image,
                   use_matplotlib=matplotlib_plotting)
        show_image(canny_edges3, "CannyC {:.2f} {:.2f}".format(low, high), save_image=save_image,
                   use_matplotlib=matplotlib_plotting)
        high += 0.2
    low += 0.2
    high = low

# 2.3
low = 0.1
high = 0.3
for i in range(3):
    canny_edges = hyst_thresh_auto(edges, high, low)
    canny_edges2 = hyst_thresh_auto(edges2, high, low)
    canny_edges3 = hyst_thresh_auto(edges3, high, low)
    show_image(canny_edges, "AutoA {:.2f} {:.2f}".format(low, high), save_image=save_image,
               use_matplotlib=matplotlib_plotting)
    show_image(canny_edges2, "AutoB {:.2f} {:.2f}".format(low, high), save_image=save_image,
               use_matplotlib=matplotlib_plotting)
    show_image(canny_edges3, "AutoC {:.2f} {:.2f}".format(low, high), save_image=save_image,
               use_matplotlib=matplotlib_plotting)
    low += 0.15
    high += 0.15

cv2.destroyAllWindows()
