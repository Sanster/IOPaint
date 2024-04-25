import os

from iopaint.tests.utils import check_device, current_dir

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pytest
import torch

from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, SDSampler
from iopaint.tests.test_model import get_config, assert_equal


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
def test_sdxl(device, strategy, sampler):
    sd_steps = check_device(device)

    model = ModelManager(
        name="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )
    cfg = get_config(
        strategy=strategy,
        prompt="face of a fox, sitting on a bench",
        sd_steps=sd_steps,
        sd_strength=1.0,
        sd_guidance_scale=7.0,
    )
    cfg.sd_sampler = sampler

    assert_equal(
        model,
        cfg,
        f"sdxl_device_{device}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=2,
        fy=2,
    )


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
def test_sdxl_cpu_text_encoder(device, strategy, sampler):
    sd_steps = check_device(device)

    model = ModelManager(
        name="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=True,
    )
    cfg = get_config(
        strategy=strategy,
        prompt="face of a fox, sitting on a bench",
        sd_steps=sd_steps,
        sd_strength=1.0,
        sd_guidance_scale=7.0,
    )
    cfg.sd_sampler = sampler

    assert_equal(
        model,
        cfg,
        f"sdxl_device_{device}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=2,
        fy=2,
    )


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize(
    "rect",
    [
        [-128, -128, 1024, 1024],
    ],
)
def test_sdxl_outpainting(device, rect):
    sd_steps = check_device(device)

    model = ModelManager(
        name="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )

    cfg = get_config(
        strategy=HDStrategy.ORIGINAL,
        prompt="a dog sitting on a bench in the park",
        sd_steps=sd_steps,
        use_extender=True,
        extender_x=rect[0],
        extender_y=rect[1],
        extender_width=rect[2],
        extender_height=rect[3],
        sd_strength=1.0,
        sd_guidance_scale=8.0,
        sd_sampler=SDSampler.ddim,
    )

    assert_equal(
        model,
        cfg,
        f"sdxl_outpainting_dog_ddim_{'_'.join(map(str, rect))}_device_{device}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1.5,
        fy=1.5,
    )
