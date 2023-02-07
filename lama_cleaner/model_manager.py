import torch
import gc

from lama_cleaner.model.fcf import FcF
from lama_cleaner.model.lama import LaMa
from lama_cleaner.model.ldm import LDM
from lama_cleaner.model.manga import Manga
from lama_cleaner.model.mat import MAT
from lama_cleaner.model.paint_by_example import PaintByExample
from lama_cleaner.model.instruct_pix2pix import InstructPix2Pix
from lama_cleaner.model.sd import SD15, SD2
from lama_cleaner.model.zits import ZITS
from lama_cleaner.model.opencv2 import OpenCV2
from lama_cleaner.schema import Config

models = {"lama": LaMa, "ldm": LDM, "zits": ZITS, "mat": MAT, "fcf": FcF, "sd1.5": SD15, "cv2": OpenCV2, "manga": Manga,
          "sd2": SD2, "paint_by_example": PaintByExample, "instruct_pix2pix": InstructPix2Pix}


class ModelManager:
    def __init__(self, name: str, device: torch.device, **kwargs):
        self.name = name
        self.device = device
        self.kwargs = kwargs
        self.model = self.init_model(name, device, **kwargs)

    def init_model(self, name: str, device, **kwargs):
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
        return self.model(image, mask, config)

    def switch(self, new_name: str):
        if new_name == self.name:
            return
        try:
            if (torch.cuda.memory_allocated() > 0):
                # Clear current loaded model from memory
                torch.cuda.empty_cache()
                del self.model
                gc.collect()

            self.model = self.init_model(new_name, self.device, **self.kwargs)
            self.name = new_name
        except NotImplementedError as e:
            raise e
