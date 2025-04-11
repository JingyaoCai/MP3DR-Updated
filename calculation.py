import numpy as np
import os
import cv2

def calculate_centers(masks):

    centers = []
    
    for i in range(masks.shape[0]):
        mask = masks[i]
        
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            print(f"Warning: mask {i+1} has no valid points")
            centers.append([-1, -1])  
            continue
        
        center_x = np.mean(x_indices)
        center_y = np.mean(y_indices)
        
        centers.append([center_x, center_y])
    
    return np.array(centers)


def get_single_depth_array(masks, depth_array):

    n, x, y = masks.shape
    single_depth_array = np.zeros_like(masks, dtype=depth_array.dtype)

    depth_min = depth_array.min()
    depth_max = depth_array.max()

    for i in range(n):
        person_depth = depth_array * (masks[i] > 0)

        if depth_max > depth_min:
            person_depth = ((person_depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

        single_depth_array[i] = person_depth

    return single_depth_array



def save_single_depth_images(single_depth_array, output_dir="SingleDepthImage"):

    os.makedirs(output_dir, exist_ok=True)
    n = single_depth_array.shape[0]

    for i in range(n):
        output_path = os.path.join(output_dir, f"human_{i+1}_depth.png")
        cv2.imwrite(output_path, single_depth_array[i])
        print(f"Saved {i+1} person depth: {output_path}")


def calculate_normalized_depth_averages(single_depth_array, depth_array):

    n = single_depth_array.shape[0]
    averages = np.zeros(n)
    normalized_averages = np.zeros(n)
    
    depth_min = depth_array.min()
    depth_max = depth_array.max()
    
    for i in range(n):
        person_depth = single_depth_array[i]
        
        valid_mask = person_depth > 0
        
        if np.any(valid_mask):
            averages[i] = np.mean(person_depth[valid_mask])
        else:
            averages[i] = 0
    
    if depth_max > depth_min:  
        normalized_averages = (averages - depth_min) / (depth_max - depth_min)
    
    return normalized_averages


def normalize_centers(centers, image_size):

    height, width = image_size
    
    normalized_centers = np.zeros_like(centers, dtype=np.float32)
    
    normalized_centers[:, 0] = centers[:, 0] / width
    
    normalized_centers[:, 1] = centers[:, 1] / height
    
    return normalized_centers


def combine_normalized_coordinates(centers, normalized_depths):

    n = len(centers)
    xyz = np.zeros((n, 3))
    
    xyz[:, :2] = centers
    
    xyz[:, 2] = normalized_depths
    
    
    return xyz