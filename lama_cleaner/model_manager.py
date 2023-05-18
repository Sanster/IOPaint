import torch
import gc

from loguru import logger

from lama_cleaner.const import SD15_MODELS
from lama_cleaner.helper import switch_mps_device
from lama_cleaner.model.controlnet import ControlNet
from lama_cleaner.model.fcf import FcF
from lama_cleaner.model.lama import LaMa
from lama_cleaner.model.ldm import LDM
from lama_cleaner.model.manga import Manga
from lama_cleaner.model.mat import MAT
from lama_cleaner.model.paint_by_example import PaintByExample
from lama_cleaner.model.instruct_pix2pix import InstructPix2Pix
from lama_cleaner.model.sd import SD15, SD2, Anything4, RealisticVision14
from lama_cleaner.model.utils import torch_gc
from lama_cleaner.model.zits import ZITS
from lama_cleaner.model.opencv2 import OpenCV2
from lama_cleaner.schema import Config

models = {
    "lama": LaMa,
    "ldm": LDM,
    "zits": ZITS,
    "mat": MAT,
    "fcf": FcF,
    SD15.name: SD15,
    Anything4.name: Anything4,
    RealisticVision14.name: RealisticVision14,
    "cv2": OpenCV2,
    "manga": Manga,
    "sd2": SD2,
    "paint_by_example": PaintByExample,
    "instruct_pix2pix": InstructPix2Pix,
}


class ModelManager:
    def __init__(self, name: str, device: torch.device, **kwargs):
        self.name = name
        self.device = device
        self.kwargs = kwargs
        self.model = self.init_model(name, device, **kwargs)

    def init_model(self, name: str, device, **kwargs):
        if name in SD15_MODELS and kwargs.get("sd_controlnet", False):
            return ControlNet(device, **{**kwargs, "name": name})

        if name in models:
            model = models[name](device, **kwargs)
        else:
            raise NotImplementedError(f"Not supported model: {name}")
        return model

    def is_downloaded(self, name: str) -> bool:
        if name in models:
            return models[name].is_downloaded()
        else:
            raise NotImplementedError(f"Not supported model: {name}")

    def __call__(self, image, mask, config: Config):
        self.switch_controlnet_method(control_method=config.controlnet_method)
        return self.model(image, mask, config)

    def switch(self, new_name: str, **kwargs):
        if new_name == self.name:
            return
        try:
            if torch.cuda.memory_allocated() > 0:
                # Clear current loaded model from memory
                torch.cuda.empty_cache()
                del self.model
                gc.collect()

            self.model = self.init_model(
                new_name, switch_mps_device(new_name, self.device), **self.kwargs
            )
            self.name = new_name
        except NotImplementedError as e:
            raise e

    def switch_controlnet_method(self, control_method: str):
        if not self.kwargs.get("sd_controlnet"):
            return
        if self.kwargs["sd_controlnet_method"] == control_method:
            return
        if self.model.is_local_sd_model:
            # is_native_control_inpaint 表示加载了普通 SD 模型
            if (
                self.model.is_native_control_inpaint
                and control_method != "control_v11p_sd15_inpaint"
            ):
                raise RuntimeError(
                    f"--sd-local-model-path load a normal SD model, "
                    f"to use {control_method} you should load an inpainting SD model"
                )
            elif (
                not self.model.is_native_control_inpaint
                and control_method == "control_v11p_sd15_inpaint"
            ):
                raise RuntimeError(
                    f"--sd-local-model-path load an inpainting SD model, "
                    f"to use {control_method} you should load a norml SD model"
                )

        del self.model
        torch_gc()

        old_method = self.kwargs["sd_controlnet_method"]
        self.kwargs["sd_controlnet_method"] = control_method
        self.model = self.init_model(
            self.name, switch_mps_device(self.name, self.device), **self.kwargs
        )
        logger.info(f"Switch ControlNet method from {old_method} to {control_method}")
