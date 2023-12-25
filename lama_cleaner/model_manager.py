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

        self.sd_controlnet = False
        self.sd_controlnet_method = ""
        self.model = self.init_model(name, device, **kwargs)

    @property
    def current_model(self) -> Dict:
        return self.available_models[name].model_dump()

    def init_model(self, name: str, device, **kwargs):
        logger.info(f"Loading model: {name}")
        if name not in self.available_models:
            raise NotImplementedError(f"Unsupported model: {name}")

        model_info = self.available_models[name]
        kwargs = {
            **kwargs,
            "model_info": model_info,
            "sd_controlnet": self.sd_controlnet,
            "sd_controlnet_method": self.sd_controlnet_method,
        }

        if model_info.model_type in [ModelType.INPAINT, ModelType.DIFFUSERS_OTHER]:
            return models[name](device, **kwargs)

        if self.sd_controlnet:
            return ControlNet(device, **kwargs)
        else:
            if model_info.model_type in [
                ModelType.DIFFUSERS_SD_INPAINT,
                ModelType.DIFFUSERS_SD,
            ]:
                return SD(device, **kwargs)

            if model_info.model_type in [
                ModelType.DIFFUSERS_SDXL_INPAINT,
                ModelType.DIFFUSERS_SDXL,
            ]:
                return SDXL(device, **kwargs)

        raise NotImplementedError(f"Unsupported model: {name}")

    def __call__(self, image, mask, config: Config):
        self.switch_controlnet_method(config)
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
        old_sd_controlnet_method = self.sd_controlnet_method
        self.name = new_name

        if (
            self.available_models[new_name].support_controlnet
            and self.sd_controlnet_method
            not in self.available_models[new_name].controlnets
        ):
            self.sd_controlnet_method = self.available_models[new_name].controlnets[0]
        try:
            # TODO: enable/disable controlnet without reload model
            del self.model
            torch_gc()

            self.model = self.init_model(
                new_name, switch_mps_device(new_name, self.device), **self.kwargs
            )
        except Exception as e:
            self.name = old_name
            self.sd_controlnet_method = old_sd_controlnet_method
            logger.info(f"Switch model from {old_name} to {new_name} failed, rollback")
            self.model = self.init_model(
                old_name, switch_mps_device(old_name, self.device), **self.kwargs
            )
            raise e

    def switch_controlnet_method(self, config):
        if not self.available_models[self.name].support_controlnet:
            return

        if (
            self.sd_controlnet
            and config.controlnet_method
            and self.sd_controlnet_method != config.controlnet_method
        ):
            old_sd_controlnet_method = self.sd_controlnet_method
            self.sd_controlnet_method = config.controlnet_method
            self.model.switch_controlnet_method(config.controlnet_method)
            logger.info(
                f"Switch Controlnet method from {old_sd_controlnet_method} to {config.controlnet_method}"
            )
        elif self.sd_controlnet != config.controlnet_enabled:
            self.sd_controlnet = config.controlnet_enabled
            self.sd_controlnet_method = config.controlnet_method

            self.model = self.init_model(
                self.name, switch_mps_device(self.name, self.device), **self.kwargs
            )
            if not config.controlnet_enabled:
                logger.info(f"Disable controlnet")
            else:
                logger.info(f"Enable controlnet: {config.controlnet_method}")

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
                if not self.model.model.get_list_adapters():
                    self.model.model.load_lora_weights(self.model.lcm_lora_id)
            else:
                self.model.model.disable_lora()
