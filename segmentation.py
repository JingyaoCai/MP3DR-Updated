from  PIL  import  Image
from lang_sam import LangSAM
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def lsam(image_path):
    model = LangSAM()
    
    image_pil = [Image.open(image_path).convert("RGB")]
    
    text_prompt = ["human"]
    
    result = model.predict(image_pil, text_prompt)
    
    masks = result[0]['masks']
    
    return masks

def refine_masks(masks, threshold=0.8, kernel_size=3, hole_area_thresh=1):
    refined_masks = []
    for i in range(masks.shape[0]):
        mask = (masks[i] > threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        h, w = closed.shape
        floodfill = closed.copy()
        temp = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(floodfill, temp, (0, 0), 255)
        holes = cv2.bitwise_not(floodfill)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes)
        small_holes = np.zeros_like(holes)

        for j in range(1, num_labels):  
            area = stats[j, cv2.CC_STAT_AREA]
            if area < hole_area_thresh:
                small_holes[labels == j] = 255

        filled = cv2.bitwise_or(closed, small_holes)
        refined_masks.append((filled > 0).astype(np.uint8))

    return np.stack(refined_masks, axis=0)


def save_humans(image_path, masks):

    output_dir = "SegmentedHuman"
    os.makedirs(output_dir, exist_ok=True)
                  
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: can't read image {image_path}")
        return

    H, W = image.shape[:2] 

    for i in range(masks.shape[0]):
        mask = masks[i]  

        segmented = np.zeros_like(image)  
        for c in range(image.shape[2]):  
            segmented[:, :, c] = image[:, :, c] * (mask > 0)  

        output_path = os.path.join(output_dir, f"human_{i+1}.png")
        cv2.imwrite(output_path, segmented)
        print(f"save: {output_path}")

def save_test_humans(image_path, masks):
    output_dir = "TestSingleHuman"
    os.makedirs(output_dir, exist_ok=True)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: can't read image {image_path}")
        return

    for i in range(len(masks)):
        mask = masks[i]  
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            print(f"Warning: mask {i+1} is empty")
            continue
            
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        cropped_image = image[y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]

        rgba_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)
        
        # alpha channel
        rgba_image[:, :, 3] = (cropped_mask * 255).astype(np.uint8)
        
        output_path = os.path.join(output_dir, f"human_{i+1}.png")
        cv2.imwrite(output_path, rgba_image)
        # print(f"Save the cropped images: {output_path}")




