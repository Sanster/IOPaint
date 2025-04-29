import argparse
import cv2
import numpy as np
import os
import random
import sys
import torch
from lama_cleaner.helper import load_img, resize_max_size
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config
from pathlib import Path


model = ModelManager("lama","cpu")

def process_image(image_path, mask_path, output_path, config):
    with open(image_path, "rb") as f:
        image_content = f.read()
    image, alpha_channel, exif = load_img(image_content, return_exif=True)

    with open(mask_path, "rb") as f:
        mask_content = f.read()
    mask, _ = load_img(mask_content, gray=True)

    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape[:2]} not equal to Image shape {image.shape[:2]}")

    original_shape = image.shape
    interpolation = cv2.INTER_CUBIC
    size_limit = max(image.shape)

    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)


    res_np_img = model(image, mask, config)

    if alpha_channel is not None:
        if alpha_channel.shape[:2] != res_np_img.shape[:2]:
            alpha_channel = cv2.resize(alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0]))
        res_np_img = np.concatenate((res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1)

    cv2.imwrite(output_path, res_np_img)

import os

def main():
    parser = argparse.ArgumentParser(description="Inpaint an image using a mask.")
    parser.add_argument("image_directory", type=str, help="Path to the input image directory.")
    parser.add_argument("mask_path", type=str, help="Path to the mask image.")
    parser.add_argument("--output_path", type=str, help="Path to save the output images.",
                        default=None)  # Set default value to None
    # Add other arguments for the Config object here
    args = parser.parse_args()

    # If output_path is not provided, use default value (a folder named 'output' inside image_directory)
    if args.output_path is None:
        args.output_path = os.path.join(args.image_directory, "output")

    config = Config(
        # Initialize the Config object with the corresponding arguments
	ldm_steps=25,
	hd_strategy='Crop',
    	hd_strategy_crop_margin=196,
     	hd_strategy_crop_trigger_size=1280,
    	hd_strategy_resize_limit=2048, 
    )

    image_directory = Path(args.image_directory)
    output_directory = Path(args.output_path)

    if not output_directory.exists():
        output_directory.mkdir(parents=True)
        
    import time

    # Get a sorted list of image paths
    sorted_image_paths = sorted(image_directory.glob("*"))
    
    # Calculate the total number of image files
    total_files = sum([1 for path in sorted_image_paths if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    
    # Initialize processed files counter
    processed_files = 0
    
    for image_path in sorted_image_paths:
        if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            start_time = time.time()
            output_image_path = output_directory / image_path.name
            process_image(str(image_path), args.mask_path, str(output_image_path), config)
    
            # Update processed files counter
            processed_files += 1
    
            # Calculate remaining files and time
            remaining_files = total_files - processed_files
            processing_time = time.time() - start_time
            remaining_time = remaining_files * processing_time / 60
    
            # Print the remaining files and time
            print(f"Remaining files: {remaining_files}, estimated time left: {remaining_time:.2f} minutes")

if __name__ == "__main__":
    main()
