import os

from iopaint.const import SD_BRUSHNET_CHOICES
from iopaint.tests.utils import check_device, get_config, assert_equal

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import pytest
import torch

from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, SDSampler, PowerPaintTask

current_dir = Path(__file__).parent.absolute().resolve()
save_dir = current_dir / "result"
save_dir.mkdir(exist_ok=True, parents=True)


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize("sampler", [SDSampler.dpm_plus_plus_2m_karras])
def test_runway_brushnet(device, sampler):
    sd_steps = check_device(device)
    model = ModelManager(
        name="runwayml/stable-diffusion-v1-5",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )
    cfg = get_config(
        strategy=HDStrategy.ORIGINAL,
        prompt="face of a fox, sitting on a bench",
        sd_steps=sd_steps,
        sd_guidance_scale=7.5,
        enable_brushnet=True,
        brushnet_method=SD_BRUSHNET_CHOICES[0],
    )
    cfg.sd_sampler = sampler

    assert_equal(
        model,
        cfg,
        f"brushnet_random_mask_{device}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("sampler", [SDSampler.dpm_plus_plus_2m])
def test_runway_powerpaint_v2(device, sampler):
    sd_steps = check_device(device)
    model = ModelManager(
        name="runwayml/stable-diffusion-v1-5",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )

    tasks = {
        PowerPaintTask.text_guided: {
            "prompt": "face of a fox, sitting on a bench",
            "scale": 7.5,
        },
        PowerPaintTask.context_aware: {
            "prompt": "face of a fox, sitting on a bench",
            "scale": 7.5,
        },
        PowerPaintTask.shape_guided: {
            "prompt": "face of a fox, sitting on a bench",
            "scale": 7.5,
        },
        PowerPaintTask.object_remove: {
            "prompt": "",
            "scale": 12,
        },
        PowerPaintTask.outpainting: {
            "prompt": "",
            "scale": 7.5,
        },
    }

    for task, data in tasks.items():
        cfg = get_config(
            strategy=HDStrategy.ORIGINAL,
            prompt=data["prompt"],
            negative_prompt="out of frame, lowres, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, watermark, signature",
            sd_steps=sd_steps,
            sd_guidance_scale=data["scale"],
            enable_powerpaint_v2=True,
            powerpaint_task=task,
            sd_sampler=sampler,
            sd_mask_blur=11,
            sd_seed=42,
            # sd_keep_unmasked_area=False
        )
        if task == PowerPaintTask.outpainting:
            cfg.use_extender = True
            cfg.extender_x = -128
            cfg.extender_y = -128
            cfg.extender_width = 768
            cfg.extender_height = 768

        assert_equal(
            model,
            cfg,
            f"powerpaint_v2_{device}_{task}.png",
            img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
            mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        )
