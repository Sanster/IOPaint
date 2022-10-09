from enum import Enum

from pydantic import BaseModel


class HDStrategy(str, Enum):
    ORIGINAL = "Original"
    RESIZE = "Resize"
    CROP = "Crop"


class LDMSampler(str, Enum):
    ddim = "ddim"
    plms = "plms"


class SDSampler(str, Enum):
    ddim = "ddim"
    pndm = "pndm"


class Config(BaseModel):
    ldm_steps: int
    ldm_sampler: str = LDMSampler.plms
    zits_wireframe: bool = True
    hd_strategy: str
    hd_strategy_crop_margin: int
    hd_strategy_crop_trigger_size: int
    hd_strategy_resize_limit: int

    prompt: str = ""
    # 始终是在原图尺度上的值
    use_croper: bool = False
    croper_x: int = None
    croper_y: int = None
    croper_height: int = None
    croper_width: int = None

    # sd
    sd_mask_blur: int = 0
    sd_strength: float = 0.75
    sd_steps: int = 50
    sd_guidance_scale: float = 7.5
    sd_sampler: str = SDSampler.ddim
    # -1 mean random seed
    sd_seed: int = 42

    # cv2
    cv2_flag: str = 'INPAINT_NS'
    cv2_radius: int = 4
