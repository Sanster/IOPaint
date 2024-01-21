import os

from iopaint.tests.utils import check_device, get_config, assert_equal

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import pytest
import torch

from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy

current_dir = Path(__file__).parent.absolute().resolve()
save_dir = current_dir / "result"
save_dir.mkdir(exist_ok=True, parents=True)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_anytext(device):
    sd_steps = check_device(device)
    model = ModelManager(
        name="Sanster/AnyText",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )

    cfg = get_config(
        strategy=HDStrategy.ORIGINAL,
        prompt='Characters written in chalk on the blackboard that says "DADDY", best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks',
        negative_prompt="low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
        sd_steps=sd_steps,
        sd_guidance_scale=9.0,
        sd_seed=66273235,
        sd_match_histograms=True
    )

    assert_equal(
        model,
        cfg,
        f"anytext.png",
        img_p=current_dir / "anytext_ref.jpg",
        mask_p=current_dir / "anytext_mask.jpg",
    )
