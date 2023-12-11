import gc
from typing import List, Dict

import torch
from loguru import logger

from lama_cleaner.download import scan_models
from lama_cleaner.helper import switch_mps_device
from lama_cleaner.model import models, ControlNet, SD, SDXL
from lama_cleaner.model.utils import torch_gc
from lama_cleaner.schema import Config, ModelInfo, ModelType


class ModelManager:
    def __init__(self, name: str, device: torch.device, **kwargs):
        self.name = name
        self.device = device
        self.kwargs = kwargs
        self.available_models: Dict[str, ModelInfo] = {}
        self.scan_models()
        self.model = self.init_model(name, device, **kwargs)

    def init_model(self, name: str, device, **kwargs):
        for old_name, model_cls in models.items():
            if name == old_name and hasattr(model_cls, "model_id_or_path"):
                name = model_cls.model_id_or_path
        if name not in self.available_models:
            raise NotImplementedError(f"Unsupported model: {name}")

        sd_controlnet_enabled = kwargs.get("sd_controlnet", False)
        model_info = self.available_models[name]
        if model_info.model_type in [ModelType.INPAINT, ModelType.DIFFUSERS_OTHER]:
            return models[name](device, **kwargs)

        if sd_controlnet_enabled:
            return ControlNet(device, **{**kwargs, "model_info": model_info})
        else:
            if model_info.model_type in [
                ModelType.DIFFUSERS_SD_INPAINT,
                ModelType.DIFFUSERS_SD,
            ]:
                return SD(device, model_id_or_path=model_info.path, **kwargs)

            if model_info.model_type in [
                ModelType.DIFFUSERS_SDXL_INPAINT,
                ModelType.DIFFUSERS_SDXL,
            ]:
                return SDXL(device, model_id_or_path=model_info.path, **kwargs)

        raise NotImplementedError(f"Unsupported model: {name}")

    def __call__(self, image, mask, config: Config):
        self.switch_controlnet_method(control_method=config.controlnet_method)
        self.enable_disable_freeu(config)
        self.enable_disable_lcm_lora(config)
        return self.model(image, mask, config)

    def scan_models(self) -> List[ModelInfo]:
        available_models = scan_models()
        self.available_models = {it.name: it for it in available_models}
        return available_models

    def switch(self, new_name: str):
        if new_name == self.name:
            return

        old_name = self.name
        self.name = new_name

        try:
            if torch.cuda.memory_allocated() > 0:
                # Clear current loaded model from memory
                torch.cuda.empty_cache()
                del self.model
                gc.collect()

            self.model = self.init_model(
                new_name, switch_mps_device(new_name, self.device), **self.kwargs
            )
        except Exception as e:
            self.name = old_name
            raise e

    def switch_controlnet_method(self, control_method: str):
        if not self.kwargs.get("sd_controlnet"):
            return
        if self.kwargs["sd_controlnet_method"] == control_method:
            return

        if not self.available_models[self.name].support_controlnet:
            return

        del self.model
        torch_gc()

        old_method = self.kwargs["sd_controlnet_method"]
        self.kwargs["sd_controlnet_method"] = control_method
        self.model = self.init_model(
            self.name, switch_mps_device(self.name, self.device), **self.kwargs
        )
        logger.info(f"Switch ControlNet method from {old_method} to {control_method}")

    def enable_disable_freeu(self, config: Config):
        if str(self.model.device) == "mps":
            return

        if self.available_models[self.name].support_freeu:
            if config.sd_freeu:
                freeu_config = config.sd_freeu_config
                self.model.model.enable_freeu(
                    s1=freeu_config.s1,
                    s2=freeu_config.s2,
                    b1=freeu_config.b1,
                    b2=freeu_config.b2,
                )
            else:
                self.model.model.disable_freeu()

    def enable_disable_lcm_lora(self, config: Config):
        if self.available_models[self.name].support_lcm_lora:
            if config.sd_lcm_lora:
                if not self.model.model.pipe.get_list_adapters():
                    self.model.model.load_lora_weights(self.model.lcm_lora_id)
            else:
                self.model.model.disable_lora()
