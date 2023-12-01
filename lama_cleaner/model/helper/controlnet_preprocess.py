import torch
import PIL
import cv2
from PIL import Image
import numpy as np


def make_canny_control_image(image: np.ndarray) -> Image:
    canny_image = cv2.Canny(image, 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = PIL.Image.fromarray(canny_image)
    control_image = canny_image
    return control_image


def make_openpose_control_image(image: np.ndarray) -> Image:
    from controlnet_aux import OpenposeDetector

    processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    control_image = processor(image, hand_and_face=True)
    return control_image


def make_depth_control_image(image: np.ndarray) -> Image:
    from transformers import pipeline

    depth_estimator = pipeline("depth-estimation")
    depth_image = depth_estimator(PIL.Image.fromarray(image))["depth"]
    depth_image = np.array(depth_image)
    depth_image = depth_image[:, :, None]
    depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)
    control_image = PIL.Image.fromarray(depth_image)
    return control_image


def make_inpaint_control_image(image: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """
    image: [H, W, C] RGB
    mask: [H, W, 1] 255 means area to repaint
    """
    image = image.astype(np.float32) / 255.0
    image[mask[:, :, -1] > 128] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image
