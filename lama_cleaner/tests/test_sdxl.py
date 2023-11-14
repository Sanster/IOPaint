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


@pytest.mark.parametrize("sd_device", ["mps"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
@pytest.mark.parametrize("cpu_textencoder", [False])
@pytest.mark.parametrize("disable_nsfw", [True])
def test_sdxl(sd_device, strategy, sampler, cpu_textencoder, disable_nsfw):
    def callback(i, t, latents):
        pass

    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    sd_steps = 20
    model = ModelManager(
        name="sdxl",
        device=torch.device(sd_device),
        hf_access_token="",
        sd_run_local=False,
        disable_nsfw=disable_nsfw,
        sd_cpu_textencoder=cpu_textencoder,
        callback=callback,
    )
    cfg = get_config(
        strategy,
        prompt="a fox sitting on a bench",
        sd_steps=sd_steps,
        sd_strength=0.99,
        sd_guidance_scale=7.0,
    )
    cfg.sd_sampler = sampler

    name = f"device_{sd_device}_{sampler}_cpu_textencoder_{cpu_textencoder}_disnsfw_{disable_nsfw}"

    assert_equal(
        model,
        cfg,
        f"sdxl_{name}.png",
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
    def callback(i, t, latents):
        pass

    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    model = ModelManager(
        name="sdxl",
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
        negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
        sd_steps=20,
        use_croper=True,
        croper_is_outpainting=True,
        croper_x=rect[0],
        croper_y=rect[1],
        croper_width=rect[2],
        croper_height=rect[3],
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
