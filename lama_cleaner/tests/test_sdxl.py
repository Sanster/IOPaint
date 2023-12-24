import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import pytest
import torch

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import HDStrategy, SDSampler, FREEUConfig
from lama_cleaner.tests.test_model import get_config, assert_equal

current_dir = Path(__file__).parent.absolute().resolve()
save_dir = current_dir / "result"
save_dir.mkdir(exist_ok=True, parents=True)


@pytest.mark.parametrize("sd_device", ["cuda", "mps"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
def test_sdxl(sd_device, strategy, sampler):
    def callback(i, t, latents):
        pass

    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    sd_steps = 20
    model = ModelManager(
        name="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        device=torch.device(sd_device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        callback=callback,
    )
    cfg = get_config(
        strategy,
        prompt="face of a fox, sitting on a bench",
        sd_steps=sd_steps,
        sd_strength=1.0,
        sd_guidance_scale=7.0,
    )
    cfg.sd_sampler = sampler

    name = f"device_{sd_device}_{sampler}"

    assert_equal(
        model,
        cfg,
        f"sdxl_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=2,
        fy=2,
    )


@pytest.mark.parametrize("sd_device", ["cuda", "mps"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
def test_sdxl_lcm_lora_and_freeu(sd_device, strategy, sampler):
    def callback(i, t, latents):
        pass

    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    sd_steps = 5
    model = ModelManager(
        name="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        device=torch.device(sd_device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        callback=callback,
    )
    cfg = get_config(
        strategy,
        prompt="face of a fox, sitting on a bench",
        sd_steps=sd_steps,
        sd_strength=1.0,
        sd_guidance_scale=2.0,
        sd_lcm_lora=True,
    )
    cfg.sd_sampler = sampler

    name = f"device_{sd_device}_{sampler}"

    assert_equal(
        model,
        cfg,
        f"sdxl_{name}_lcm_lora.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=2,
        fy=2,
    )

    cfg = get_config(
        strategy,
        prompt="face of a fox, sitting on a bench",
        sd_steps=sd_steps,
        sd_guidance_scale=7.5,
        sd_freeu=True,
        sd_freeu_config=FREEUConfig(),
    )

    assert_equal(
        model,
        cfg,
        f"sdxl_{name}_freeu.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=2,
        fy=2,
    )


@pytest.mark.parametrize("sd_device", ["mps"])
@pytest.mark.parametrize(
    "rect",
    [
        [-128, -128, 1024, 1024],
    ],
)
def test_sdxl_outpainting(sd_device, rect):
    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    model = ModelManager(
        name="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        device=torch.device(sd_device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )

    cfg = get_config(
        HDStrategy.ORIGINAL,
        prompt="a dog sitting on a bench in the park",
        sd_steps=20,
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
        f"sdxl_outpainting_dog_ddim_{'_'.join(map(str, rect))}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1.5,
        fy=1.5,
    )
