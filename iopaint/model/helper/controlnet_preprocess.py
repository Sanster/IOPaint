import torch
import PIL
import cv2
from PIL import Image
import numpy as np

from iopaint.helper import pad_img_to_modulo


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


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(
        input_image,
        (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
    )
    return img


def make_depth_control_image(image: np.ndarray) -> Image:
    from controlnet_aux import MidasDetector

    midas = MidasDetector.from_pretrained("lllyasviel/Annotators")

    origin_height, origin_width = image.shape[:2]
    pad_image = pad_img_to_modulo(image, mod=64, square=False, min_size=512)
    depth_image = midas(pad_image)
    depth_image = depth_image[0:origin_height, 0:origin_width]
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
