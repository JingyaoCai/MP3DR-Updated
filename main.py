import argparse
from PIL import Image
from lang_sam import LangSAM
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from mpl_toolkits.mplot3d import Axes3D 

# segmentation.py import
from segmentation import lsam, refine_masks, save_humans, save_test_humans

# depthestimation.py import
from depthestimation import depthv2, save_depth_image, extract_depth_from_masks

# calculation.py import
from calculation import (
    calculate_centers,
    get_single_depth_array,
    save_single_depth_images,
    calculate_normalized_depth_averages,
    normalize_centers,
    combine_normalized_coordinates
)

# display.py import
from display import plot_images_low_res, plot_images_high_res, result_high_res_save

def main(image_path):
    original_image = cv2.imread(image_path)
    image_size = (original_image.shape[:2])

    # Segmentation
    masks = lsam(image_path)

    # Refine Masks
    masks = refine_masks(masks)

    # Save Alpha Single Human Image for Test
    save_test_humans(image_path, masks) 

    # Depth Estimation
    depth_array = depthv2(image_path)

    # Single Depth Array
    single_depth_array = get_single_depth_array(masks, depth_array)

    # Normalized Depth Averages
    normalized_depths = calculate_normalized_depth_averages(single_depth_array, depth_array)

    # Calculate Centers
    centers = calculate_centers(masks)

    # Combine Normalized Coordinates
    xyz = combine_normalized_coordinates(centers, normalized_depths)
    print(xyz)

    # Save Result
    result_high_res_save(xyz, "TestSingleHuman", image_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image segmentation and depth estimation on a given image.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    args = parser.parse_args()

    main(args.image_path)


# In terminal: python main.py ./MultiImages/2.jpg