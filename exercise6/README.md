# Open Challenge

The goal of this exercise is the implementation of pipeline capable of detecting objects in RGBD images. Please refer to
the documentation `report.pdf` for a detailed explanation of the implemented approach.

## File structure
- `main.py` contains the basic object recognition pipeline and call functions from the other files where necessary.
- `points_to_image.py` contains functions to create an image from a point cloud
- `cluster_matching.py` contains functions related to clustered images
- `util.py` contains helper functions which are used for image and point cloud plotting.


## How to run the code
Just run the `main.py` files. Only packages used in the previous exercises are required. Define the input point cloud
using the `input_image` variable and enable detailed debugging output (which was used to generate the figures in the
report) by setting the `debug` variable.

Each major step of the detection pipeline has its parameters in the `main.py` file just before the respective code.

### Extensions
By default, the RGB SIFT extension is enabled, and the distance weighted match counting is disabled. Change the
variables `use_colour_sift` and `use_distance_weights` in line 138 and 139 of the `main.py` file to alter this
behaviour.
    