# Open Challenge

The goal of this exercise is the implementation of pipeline capable of detecting objects in RGBD images. Please refer to
the documentation `/report/report.pdf` for a detailed explanation of the implemented approach.

## File structure
- `main.py` contains the basic object recognition pipeline and call functions from the other files where necessary.
- `points_to_image.py` contains functions to create an image from a point cloud
- `cluster_matching.py` contains functions related to clustered images
- `util.py` contains helper functions which are used for image and point cloud plotting.


## How to run the code
Just run the `main.py` files. Only packages used in the previous exercises are required.