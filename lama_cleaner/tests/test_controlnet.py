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


@pytest.mark.parametrize("sd_device", ["cuda", "mps"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.uni_pc])
@pytest.mark.parametrize("cpu_textencoder", [True])
@pytest.mark.parametrize("disable_nsfw", [True])
def test_runway_sd_1_5(sd_device, strategy, sampler, cpu_textencoder, disable_nsfw):
    if sd_device == "cuda" and not torch.cuda.is_available():
        return
    if device == "mps" and not torch.backends.mps.is_available():
        return

    sd_steps = 1 if sd_device == "cpu" else 30
    model = ModelManager(
        name="sd1.5",
        sd_controlnet=True,
        device=torch.device(sd_device),
        hf_access_token="",
        sd_run_local=True,
        disable_nsfw=disable_nsfw,
        sd_cpu_textencoder=cpu_textencoder,
    )
    cfg = get_config(strategy, prompt="a fox sitting on a bench", sd_steps=sd_steps)
    cfg.sd_sampler = sampler

    name = f"device_{sd_device}_{sampler}_cpu_textencoder_{cpu_textencoder}_disnsfw_{disable_nsfw}"

    assert_equal(
        model,
        cfg,
        f"sd_controlnet_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1.2,
        fy=1.2,
    )


@pytest.mark.parametrize("sd_device", ["cuda", "mps"])
@pytest.mark.parametrize("sampler", [SDSampler.uni_pc])
def test_local_file_path(sd_device, sampler):
    if sd_device == "cuda" and not torch.cuda.is_available():
        return
    if device == "mps" and not torch.backends.mps.is_available():
        return

    sd_steps = 1 if sd_device == "cpu" else 30
    model = ModelManager(
        name="sd1.5",
        sd_controlnet=True,
        device=torch.device(sd_device),
        hf_access_token="",
        sd_run_local=True,
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        cpu_offload=True,
        sd_local_model_path="/Users/cwq/data/models/sd-v1-5-inpainting.ckpt",
    )
    cfg = get_config(
        HDStrategy.ORIGINAL,
        prompt="a fox sitting on a bench",
        sd_steps=sd_steps,
    )
    cfg.sd_sampler = sampler

    name = f"device_{sd_device}_{sampler}"

    assert_equal(
        model,
        cfg,
        f"sd_controlnet_local_model_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )
