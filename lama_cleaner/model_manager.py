from lama_cleaner.model.fcf import FcF
from lama_cleaner.model.lama import LaMa
from lama_cleaner.model.ldm import LDM
from lama_cleaner.model.mat import MAT
from lama_cleaner.model.sd import SD14
from lama_cleaner.model.zits import ZITS
from lama_cleaner.model.opencv2 import OpenCV2
from lama_cleaner.schema import Config

models = {"lama": LaMa, "ldm": LDM, "zits": ZITS, "mat": MAT, "fcf": FcF, "sd1.4": SD14, "cv2": OpenCV2}


class ModelManager:
    def __init__(self, name: str, device, **kwargs):
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
            self.model = self.init_model(new_name, self.device, **self.kwargs)
            self.name = new_name
        except NotImplementedError as e:
            raise e
