import os

from iopaint.tests.utils import current_dir, check_device

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pytest
import torch

from iopaint.model_manager import ModelManager
from iopaint.schema import SDSampler
from iopaint.tests.test_model import get_config, assert_equal


@pytest.mark.parametrize("name", ["runwayml/stable-diffusion-inpainting"])
@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize(
    "rect",
    [
        [0, -100, 512, 512 - 128 + 100],
        [0, 128, 512, 512 - 128 + 100],
        [128, 0, 512 - 128 + 100, 512],
        [-100, 0, 512 - 128 + 100, 512],
        [0, 0, 512, 512 + 200],
        [256, 0, 512 + 200, 512],
        [-100, -100, 512 + 200, 512 + 200],
    ],
)
def test_outpainting(name, device, rect):
    sd_steps = check_device(device)

    model = ModelManager(
        name=name,
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )
    cfg = get_config(
        prompt="a dog sitting on a bench in the park",
        sd_steps=sd_steps,
        use_extender=True,
        extender_x=rect[0],
        extender_y=rect[1],
        extender_width=rect[2],
        extender_height=rect[3],
        sd_guidance_scale=8.0,
        sd_sampler=SDSampler.dpm_plus_plus_2m,
    )

    assert_equal(
        model,
        cfg,
        f"{name.replace('/', '--')}_outpainting_{'_'.join(map(str, rect))}_device_{device}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("name", ["kandinsky-community/kandinsky-2-2-decoder-inpaint"])
@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize(
    "rect",
    [
        [-128, -128, 768, 768],
    ],
)
def test_kandinsky_outpainting(name, device, rect):
    sd_steps = check_device(device)

    model = ModelManager(
        name=name,
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )
    cfg = get_config(
        prompt="a cat",
        negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
        sd_steps=sd_steps,
        use_extender=True,
        extender_x=rect[0],
        extender_y=rect[1],
        extender_width=rect[2],
        extender_height=rect[3],
        sd_guidance_scale=7,
        sd_sampler=SDSampler.dpm_plus_plus_2m,
    )

    assert_equal(
        model,
        cfg,
        f"{name.replace('/', '--')}_outpainting_{'_'.join(map(str, rect))}_device_{device}.png",
        img_p=current_dir / "cat.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1,
        fy=1,
    )


@pytest.mark.parametrize("name", ["Sanster/PowerPaint-V1-stable-diffusion-inpainting"])
@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize(
    "rect",
    [
        [-100, -100, 512 + 200, 512 + 200],
    ],
)
def test_powerpaint_outpainting(name, device, rect):
    sd_steps = check_device(device)

    model = ModelManager(
        name=name,
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        low_mem=True,
    )
    cfg = get_config(
        prompt="a dog sitting on a bench in the park",
        sd_steps=sd_steps,
        use_extender=True,
        extender_x=rect[0],
        extender_y=rect[1],
        extender_width=rect[2],
        extender_height=rect[3],
        sd_guidance_scale=8.0,
        sd_sampler=SDSampler.dpm_plus_plus_2m,
        powerpaint_task="outpainting",
    )

    assert_equal(
        model,
        cfg,
        f"{name.replace('/', '--')}_outpainting_{'_'.join(map(str, rect))}_device_{device}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )
