from typing import Optional
from enum import Enum

from PIL.Image import Image
from pydantic import BaseModel


class HDStrategy(str, Enum):
    # Use original image size
    ORIGINAL = "Original"
    # Resize the longer side of the image to a specific size(hd_strategy_resize_limit),
    # then do inpainting on the resized image. Finally, resize the inpainting result to the original size.
    # The area outside the mask will not lose quality.
    RESIZE = "Resize"
    # Crop masking area(with a margin controlled by hd_strategy_crop_margin) from the original image to do inpainting
    CROP = "Crop"


class LDMSampler(str, Enum):
    ddim = "ddim"
    plms = "plms"


class SDSampler(str, Enum):
    ddim = "ddim"
    pndm = "pndm"
    k_lms = "k_lms"
    k_euler = "k_euler"
    k_euler_a = "k_euler_a"
    dpm_plus_plus = "dpm++"
    uni_pc = "uni_pc"


class Config(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    # Configs for ldm model
    ldm_steps: int
    ldm_sampler: str = LDMSampler.plms

    # Configs for zits model
    zits_wireframe: bool = True

    # Configs for High Resolution Strategy(different way to preprocess image)
    hd_strategy: str  # See HDStrategy Enum
    hd_strategy_crop_margin: int
    # If the longer side of the image is larger than this value, use crop strategy
    hd_strategy_crop_trigger_size: int
    hd_strategy_resize_limit: int

    # Configs for Stable Diffusion 1.5
    prompt: str = ""
    negative_prompt: str = ""
    # Crop image to this size before doing sd inpainting
    # The value is always on the original image scale
    use_croper: bool = False
    croper_x: int = None
    croper_y: int = None
    croper_height: int = None
    croper_width: int = None

    # Resize the image before doing sd inpainting, the area outside the mask will not lose quality.
    # Used by sd models and paint_by_example model
    sd_scale: float = 1.0
    # Blur the edge of mask area. The higher the number the smoother blend with the original image
    sd_mask_blur: int = 0
    # Ignore this value, it's useless for inpainting
    sd_strength: float = 0.75
    # The number of denoising steps. More denoising steps usually lead to a
    # higher quality image at the expense of slower inference.
    sd_steps: int = 50
    # Higher guidance scale encourages to generate images that are closely linked
    # to the text prompt, usually at the expense of lower image quality.
    sd_guidance_scale: float = 7.5
    sd_sampler: str = SDSampler.uni_pc
    # -1 mean random seed
    sd_seed: int = 42
    sd_match_histograms: bool = False

    # Configs for opencv inpainting
    # opencv document https://docs.opencv.org/4.6.0/d7/d8b/group__photo__inpaint.html#gga8002a65f5a3328fbf15df81b842d3c3ca05e763003a805e6c11c673a9f4ba7d07
    cv2_flag: str = "INPAINT_NS"
    cv2_radius: int = 4

    # Paint by Example
    paint_by_example_steps: int = 50
    paint_by_example_guidance_scale: float = 7.5
    paint_by_example_mask_blur: int = 0
    paint_by_example_seed: int = 42
    paint_by_example_match_histograms: bool = False
    paint_by_example_example_image: Optional[Image] = None

    # InstructPix2Pix
    p2p_steps: int = 50
    p2p_image_guidance_scale: float = 1.5
    p2p_guidance_scale: float = 7.5

    # ControlNet
    controlnet_conditioning_scale: float = 0.4
    controlnet_method: str = "control_v11p_sd15_canny"
