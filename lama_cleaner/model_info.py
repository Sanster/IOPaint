from enum import Enum
from typing import List

from pydantic import computed_field, BaseModel

from lama_cleaner.const import (
    SDXL_CONTROLNET_CHOICES,
    SD2_CONTROLNET_CHOICES,
    SD_CONTROLNET_CHOICES,
)
from lama_cleaner.model import InstructPix2Pix, Kandinsky22, PowerPaint, SD2
from lama_cleaner.schema import ModelType


class ModelInfo(BaseModel):
    name: str
    path: str
    model_type: ModelType
    is_single_file_diffusers: bool = False

    @computed_field
    @property
    def need_prompt(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ] or self.name in [
            InstructPix2Pix.name,
            Kandinsky22.name,
            PowerPaint.name,
        ]

    @computed_field
    @property
    def controlnets(self) -> List[str]:
        if self.model_type in [
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ]:
            return SDXL_CONTROLNET_CHOICES
        if self.model_type in [ModelType.DIFFUSERS_SD, ModelType.DIFFUSERS_SD_INPAINT]:
            if self.name in [SD2.name]:
                return SD2_CONTROLNET_CHOICES
            else:
                return SD_CONTROLNET_CHOICES
        if self.name == PowerPaint.name:
            return SD_CONTROLNET_CHOICES
        return []

    @computed_field
    @property
    def support_strength(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ]

    @computed_field
    @property
    def support_outpainting(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ] or self.name in [Kandinsky22.name, PowerPaint.name]

    @computed_field
    @property
    def support_lcm_lora(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ]

    @computed_field
    @property
    def support_controlnet(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ] or self.name in [PowerPaint.name]

    @computed_field
    @property
    def support_freeu(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ] or self.name in [InstructPix2Pix.name]
