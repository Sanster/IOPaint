import logging
import os

from lama_cleaner.schema import Config

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

from lama_cleaner.model_manager import ModelManager


def test_model_switch():
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        sd_controlnet=True,
        sd_controlnet_method="lllyasviel/control_v11p_sd15_canny",
        device=torch.device("mps"),
        disable_nsfw=True,
        sd_cpu_textencoder=True,
        cpu_offload=False,
        callback=None,
    )

    model.switch("lama")


def test_controlnet_switch_onoff(caplog):
    name = "runwayml/stable-diffusion-inpainting"
    model = ModelManager(
        name=name,
        sd_controlnet=True,
        sd_controlnet_method="lllyasviel/control_v11p_sd15_canny",
        device=torch.device("mps"),
        disable_nsfw=True,
        sd_cpu_textencoder=True,
        cpu_offload=False,
        callback=None,
    )

    model.switch_controlnet_method(
        Config(
            name=name,
            controlnet_enabled=False,
        )
    )

    assert "Disable controlnet" in caplog.text


def test_controlnet_switch_method(caplog):
    name = "runwayml/stable-diffusion-inpainting"
    old_method = "lllyasviel/control_v11p_sd15_canny"
    new_method = "lllyasviel/control_v11p_sd15_openpose"
    model = ModelManager(
        name=name,
        sd_controlnet=True,
        sd_controlnet_method=old_method,
        device=torch.device("mps"),
        disable_nsfw=True,
        sd_cpu_textencoder=True,
        cpu_offload=False,
        callback=None,
    )

    model.switch_controlnet_method(
        Config(
            name=name,
            controlnet_enabled=True,
            controlnet_method=new_method,
        )
    )

    assert f"Switch Controlnet method from {old_method} to {new_method}" in caplog.text
