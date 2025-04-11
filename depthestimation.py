from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import cv2
import os


def depthv2(image_path):
    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    
    image = Image.open(image_path)
    
    inputs = image_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_depth = outputs.predicted_depth
    
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    
    output = prediction.squeeze().cpu().numpy()
    depth_array = (output - output.min()) / (output.max() - output.min()) * 255
    
    return depth_array.astype(np.uint8)


def save_depth_image(depth_array, image_path):


    output_dir = "DepthImage"
    os.makedirs(output_dir, exist_ok=True)
    

    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    # Save depth image
    output_path = os.path.join(output_dir, f"{name}_depth{ext}")
    cv2.imwrite(output_path, depth_array)
    print(f"Save depth image: {output_path}")


def extract_depth_from_masks(depth_path, masks):

    depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth_image is None:
        print(f"error, can't read depth image {depth_path}")
        return
    
    output_dir = "DepthCompareHuman"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(masks.shape[0]):
        mask = masks[i]
        
        segmented_depth = np.zeros_like(depth_image)
        
        segmented_depth[mask > 0] = depth_image[mask > 0]
        
        output_path = os.path.join(output_dir, f"person_{i+1}_depth.png")
        cv2.imwrite(output_path, segmented_depth)
        print(f"save {i+1} person depth: {output_path}")