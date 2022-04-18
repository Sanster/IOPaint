from enum import Enum

from pydantic import BaseModel


class HDStrategy(str, Enum):
    ORIGINAL = 'Original'
    RESIZE = 'Resize'
    CROP = 'Crop'


class Config(BaseModel):
    ldm_steps: int
    hd_strategy: str
    hd_strategy_crop_margin: int
    hd_strategy_crop_trigger_size: int
    hd_strategy_resize_limit: int
