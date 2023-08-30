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


@pytest.mark.parametrize("name", ["sd1.5"])
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
def test_outpainting(name, sd_device, rect):
    def callback(i, t, latents):
        pass

    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    model = ModelManager(
        name=name,
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
        sd_steps=50,
        use_croper=True,
        croper_is_outpainting=True,
        croper_x=rect[0],
        croper_y=rect[1],
        croper_width=rect[2],
        croper_height=rect[3],
        sd_guidance_scale=4,
        sd_sampler=SDSampler.dpm_plus_plus,
    )

    assert_equal(
        model,
        cfg,
        f"{name.replace('.', '_')}_outpainting_dpm++_{'_'.join(map(str, rect))}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("name", ["kandinsky2.2"])
@pytest.mark.parametrize("sd_device", ["mps"])
@pytest.mark.parametrize(
    "rect",
    [
        [-100, -100, 768, 768],
    ],
)
def test_kandinsky_outpainting(name, sd_device, rect):
    def callback(i, t, latents):
        pass

    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    model = ModelManager(
        name=name,
        device=torch.device(sd_device),
        hf_access_token="",
        sd_run_local=True,
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        callback=callback,
    )
    cfg = get_config(
        HDStrategy.ORIGINAL,
        prompt="a cat",
        negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
        sd_steps=50,
        use_croper=True,
        croper_is_outpainting=True,
        croper_x=rect[0],
        croper_y=rect[1],
        croper_width=rect[2],
        croper_height=rect[3],
        sd_guidance_scale=4,
        sd_sampler=SDSampler.dpm_plus_plus,
    )

    assert_equal(
        model,
        cfg,
        f"{name.replace('.', '_')}_outpainting_dpm++_{'_'.join(map(str, rect))}.png",
        img_p=current_dir / "cat.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )
