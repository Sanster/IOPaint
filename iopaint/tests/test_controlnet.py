import os

from iopaint.const import SD_CONTROLNET_CHOICES
from iopaint.tests.utils import current_dir, check_device, get_config, assert_equal

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import pytest
import torch

from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, SDSampler


model_name = "runwayml/stable-diffusion-inpainting"


def convert_controlnet_method_name(name):
    return name.replace("/", "--")


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize("controlnet_method", [SD_CONTROLNET_CHOICES[0]])
def test_runway_sd_1_5(device, controlnet_method):
    sd_steps = check_device(device)

    model = ModelManager(
        name=model_name,
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=device == "cuda",
        enable_controlnet=True,
        controlnet_method=controlnet_method,
    )

    cfg = get_config(
        prompt="a fox sitting on a bench",
        sd_steps=sd_steps,
        enable_controlnet=True,
        controlnet_conditioning_scale=0.5,
        controlnet_method=controlnet_method,
    )
    name = f"device_{device}"

    assert_equal(
        model,
        cfg,
        f"sd_controlnet_{convert_controlnet_method_name(controlnet_method)}_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_controlnet_switch(device):
    sd_steps = check_device(device)
    model = ModelManager(
        name=model_name,
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        cpu_offload=True,
        enable_controlnet=True,
        controlnet_method="lllyasviel/control_v11p_sd15_canny",
    )
    cfg = get_config(
        prompt="a fox sitting on a bench",
        sd_steps=sd_steps,
        enable_controlnet=True,
        controlnet_method="lllyasviel/control_v11f1p_sd15_depth",
    )

    assert_equal(
        model,
        cfg,
        f"controlnet_switch_canny_to_depth_device_{device}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1.2
    )


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize(
    "local_file", ["sd-v1-5-inpainting.ckpt", "v1-5-pruned-emaonly.safetensors"]
)
def test_local_file_path(device, local_file):
    sd_steps = check_device(device)

    controlnet_kwargs = dict(
        enable_controlnet=True,
        controlnet_method=SD_CONTROLNET_CHOICES[0],
    )

    model = ModelManager(
        name=local_file,
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        cpu_offload=True,
        **controlnet_kwargs,
    )
    cfg = get_config(
        prompt="a fox sitting on a bench",
        sd_steps=sd_steps,
        **controlnet_kwargs,
    )

    name = f"device_{device}"

    assert_equal(
        model,
        cfg,
        f"{convert_controlnet_method_name(controlnet_kwargs['controlnet_method'])}_local_model_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )
