import os

from loguru import logger

from iopaint.tests.utils import check_device, get_config, assert_equal, current_dir

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pytest
import torch

from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, SDSampler


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_runway_sd_1_5_low_mem(device):
    sd_steps = check_device(device)
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        low_mem=True,
    )

    all_samplers = [member.value for member in SDSampler.__members__.values()]
    print(all_samplers)
    cfg = get_config(
        strategy=HDStrategy.ORIGINAL,
        prompt="a fox sitting on a bench",
        sd_steps=sd_steps,
        sd_sampler=SDSampler.ddim,
    )

    name = f"device_{device}"

    assert_equal(
        model,
        cfg,
        f"runway_sd_{name}_low_mem.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize("sampler", [SDSampler.lcm])
def test_runway_sd_lcm_lora_low_mem(device, sampler):
    check_device(device)

    sd_steps = 5
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        low_mem=True,
    )
    cfg = get_config(
        strategy=HDStrategy.ORIGINAL,
        prompt="face of a fox, sitting on a bench",
        sd_steps=sd_steps,
        sd_guidance_scale=2,
        sd_lcm_lora=True,
    )
    cfg.sd_sampler = sampler

    assert_equal(
        model,
        cfg,
        f"runway_sd_1_5_lcm_lora_device_{device}_low_mem.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )



@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
def test_runway_norm_sd_model(device, strategy, sampler):
    sd_steps = check_device(device)
    model = ModelManager(
        name="runwayml/stable-diffusion-v1-5",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        low_mem=True,
    )
    cfg = get_config(
        strategy=strategy, prompt="face of a fox, sitting on a bench", sd_steps=sd_steps
    )
    cfg.sd_sampler = sampler

    assert_equal(
        model,
        cfg,
        f"runway_{device}_norm_sd_model_device_{device}_low_mem.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )
