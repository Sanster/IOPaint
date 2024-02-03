import os

from iopaint.schema import InpaintRequest

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

from iopaint.model_manager import ModelManager


def test_model_switch():
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        enable_controlnet=True,
        controlnet_method="lllyasviel/control_v11p_sd15_canny",
        device=torch.device("mps"),
        disable_nsfw=True,
        sd_cpu_textencoder=True,
        cpu_offload=False,
    )

    model.switch("lama")


def test_controlnet_switch_onoff(caplog):
    name = "runwayml/stable-diffusion-inpainting"
    model = ModelManager(
        name=name,
        enable_controlnet=True,
        controlnet_method="lllyasviel/control_v11p_sd15_canny",
        device=torch.device("mps"),
        disable_nsfw=True,
        sd_cpu_textencoder=True,
        cpu_offload=False,
    )

    model.switch_controlnet_method(
        InpaintRequest(
            name=name,
            enable_controlnet=False,
        )
    )

    assert "Disable controlnet" in caplog.text


def test_switch_controlnet_method(caplog):
    name = "runwayml/stable-diffusion-inpainting"
    old_method = "lllyasviel/control_v11p_sd15_canny"
    new_method = "lllyasviel/control_v11p_sd15_openpose"
    model = ModelManager(
        name=name,
        enable_controlnet=True,
        controlnet_method=old_method,
        device=torch.device("mps"),
        disable_nsfw=True,
        sd_cpu_textencoder=True,
        cpu_offload=False,
    )

    model.switch_controlnet_method(
        InpaintRequest(
            name=name,
            enable_controlnet=True,
            controlnet_method=new_method,
        )
    )

    assert f"Switch Controlnet method from {old_method} to {new_method}" in caplog.text
