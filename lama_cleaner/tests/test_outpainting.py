import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import pytest
import torch

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import HDStrategy, SDSampler
from lama_cleaner.tests.test_model import get_config, assert_equal

current_dir = Path(__file__).parent.absolute().resolve()
save_dir = current_dir / "result"
save_dir.mkdir(exist_ok=True, parents=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


@pytest.mark.parametrize("sd_device", ["mps"])
@pytest.mark.parametrize(
    "rect",
    [
        [0, -100, 512, 512 - 128 + 100],
        [0, 128, 512, 512 - 128 + 100],
        [128, 0, 512 - 128 + 100, 512],
        [-100, 0, 512 - 128 + 100, 512],
        [0, 0, 512, 512 + 200],
        [0, 0, 512 + 200, 512],
        [-100, -100, 512 + 200, 512 + 200],
    ],
)
def test_sdxl_outpainting(sd_device, rect):
    def callback(i, t, latents):
        pass

    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    sd_steps = 50 if sd_device == "cuda" else 1
    model = ModelManager(
        name="sd1.5",
        device=torch.device(sd_device),
        hf_access_token="",
        sd_run_local=True,
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        callback=callback,
    )
    cfg = get_config(
        HDStrategy.ORIGINAL,
        prompt="a dog sitting on a bench in the park",
        sd_steps=30,
        use_croper=True,
        croper_is_outpainting=True,
        croper_x=rect[0],
        croper_y=rect[1],
        croper_width=rect[2],
        croper_height=rect[3],
        sd_guidance_scale=14,
        sd_sampler=SDSampler.dpm_plus_plus,
    )

    assert_equal(
        model,
        cfg,
        f"sd15_outpainting_dpm++_{'_'.join(map(str, rect))}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )
