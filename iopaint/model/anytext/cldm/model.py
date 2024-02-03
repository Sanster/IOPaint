import os
import torch

from omegaconf import OmegaConf
from iopaint.model.anytext.ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get("state_dict", d)


def load_state_dict(ckpt_path, location="cpu"):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch

        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(
            torch.load(ckpt_path, map_location=torch.device(location))
        )
    state_dict = get_state_dict(state_dict)
    print(f"Loaded state_dict from [{ckpt_path}]")
    return state_dict


def create_model(config_path, device, cond_stage_path=None, use_fp16=False):
    config = OmegaConf.load(config_path)
    # if cond_stage_path:
    #     config.model.params.cond_stage_config.params.version = (
    #         cond_stage_path  # use pre-downloaded ckpts, in case blocked
    #     )
    config.model.params.cond_stage_config.params.device = str(device)
    if use_fp16:
        config.model.params.use_fp16 = True
        config.model.params.control_stage_config.params.use_fp16 = True
        config.model.params.unet_config.params.use_fp16 = True
    model = instantiate_from_config(config.model).cpu()
    print(f"Loaded model config from [{config_path}]")
    return model
