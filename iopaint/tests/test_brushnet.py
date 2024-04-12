import os

from iopaint.const import SD_BRUSHNET_CHOICES
from iopaint.tests.utils import check_device, get_config, assert_equal

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import pytest
import torch

from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, SDSampler, FREEUConfig

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
        sd_freeu=True,
        sd_freeu_config=FREEUConfig(),
        enable_brushnet=True,
        brushnet_method=SD_BRUSHNET_CHOICES[0]
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
@pytest.mark.parametrize("sampler", [SDSampler.dpm_plus_plus_2m_karras])
@pytest.mark.parametrize(
    "name",
    [
        "v1-5-pruned-emaonly.safetensors",
    ],
)
def test_brushnet_local_file_path(device, sampler, name):
    sd_steps = check_device(device)
    model = ModelManager(
        name=name,
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        cpu_offload=False,
    )
    cfg = get_config(
        strategy=HDStrategy.ORIGINAL,
        prompt="face of a fox, sitting on a bench",
        sd_steps=sd_steps,
        sd_seed=1234,
        enable_brushnet=True,
        brushnet_method=SD_BRUSHNET_CHOICES[1]
    )
    cfg.sd_sampler = sampler

    assert_equal(
        model,
        cfg,
        f"brushnet_segmentation_mask_{device}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1,
        fy=1,
    )
