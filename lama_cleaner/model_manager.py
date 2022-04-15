from lama_cleaner.model.lama import LaMa
from lama_cleaner.model.ldm import LDM
from lama_cleaner.schema import Config


class ModelManager:
    LAMA = 'lama'
    LDM = 'ldm'

    def __init__(self, name: str, device):
        self.name = name
        self.device = device
        self.model = self.init_model(name, device)

    def init_model(self, name: str, device):
        if name == self.LAMA:
            model = LaMa(device)
        elif name == self.LDM:
            model = LDM(device)
        else:
            raise NotImplementedError(f"Not supported model: {name}")
        return model

    def __call__(self, image, mask, config: Config):
        return self.model(image, mask, config)

    def switch(self, new_name: str):
        if new_name == self.name:
            return
        try:
            self.model = self.init_model(new_name, self.device)
            self.name = new_name
        except NotImplementedError as e:
            raise e
