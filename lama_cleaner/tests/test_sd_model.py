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
@pytest.mark.parametrize(
    "sampler",
    [
        SDSampler.ddim,
        SDSampler.pndm,
        SDSampler.k_lms,
        SDSampler.k_euler,
        SDSampler.k_euler_a,
        SDSampler.lcm,
    ],
)
def test_runway_sd_1_5_all_samplers(
    sd_device,
    sampler,
):
    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    sd_steps = 30
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        device=torch.device(sd_device),
        hf_access_token="",
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )
    cfg = get_config(
        HDStrategy.ORIGINAL, prompt="a fox sitting on a bench", sd_steps=sd_steps
    )
    cfg.sd_sampler = sampler

    name = f"device_{sd_device}_{sampler}"

    assert_equal(
        model,
        cfg,
        f"runway_sd_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("sd_device", ["cuda", "mps"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.lcm])
def test_runway_sd_lcm_lora(sd_device, strategy, sampler):
    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    sd_steps = 5
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        device=torch.device(sd_device),
        hf_access_token="",
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )
    cfg = get_config(
        strategy,
        prompt="face of a fox, sitting on a bench",
        sd_steps=sd_steps,
        sd_guidance_scale=2,
        sd_lcm_lora=True,
    )
    cfg.sd_sampler = sampler

    assert_equal(
        model,
        cfg,
        f"runway_sd_1_5_lcm_lora.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("sd_device", ["cuda", "mps"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
def test_runway_sd_freeu(sd_device, strategy, sampler):
    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    sd_steps = 30
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        device=torch.device(sd_device),
        hf_access_token="",
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )
    cfg = get_config(
        strategy,
        prompt="face of a fox, sitting on a bench",
        sd_steps=sd_steps,
        sd_guidance_scale=7.5,
        sd_freeu=True,
        sd_freeu_config=FREEUConfig(),
    )
    cfg.sd_sampler = sampler

    assert_equal(
        model,
        cfg,
        f"runway_sd_1_5_freeu.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("sd_device", ["cuda", "mps"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
def test_runway_sd_sd_strength(sd_device, strategy, sampler):
    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    sd_steps = 30
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        device=torch.device(sd_device),
        hf_access_token="",
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )
    cfg = get_config(
        strategy, prompt="a fox sitting on a bench", sd_steps=sd_steps, sd_strength=0.8
    )
    cfg.sd_sampler = sampler

    assert_equal(
        model,
        cfg,
        f"runway_sd_strength_0.8.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("sd_device", ["cuda", "mps"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
def test_runway_norm_sd_model(sd_device, strategy, sampler):
    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    sd_steps = 30
    model = ModelManager(
        name="runwayml/stable-diffusion-v1-5",
        device=torch.device(sd_device),
        hf_access_token="",
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )
    cfg = get_config(strategy, prompt="face of a fox, sitting on a bench", sd_steps=sd_steps)
    cfg.sd_sampler = sampler

    assert_equal(
        model,
        cfg,
        f"runway_{sd_device}_norm_sd_model.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("sd_device", ["cuda"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.k_euler_a])
def test_runway_sd_1_5_cpu_offload(sd_device, strategy, sampler):
    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    sd_steps = 30
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        device=torch.device(sd_device),
        hf_access_token="",
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        cpu_offload=True,
    )
    cfg = get_config(strategy, prompt="a fox sitting on a bench", sd_steps=sd_steps)
    cfg.sd_sampler = sampler

    name = f"device_{sd_device}_{sampler}"

    assert_equal(
        model,
        cfg,
        f"runway_sd_{strategy.capitalize()}_{name}_cpu_offload.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("sd_device", ["cuda", "mps"])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
@pytest.mark.parametrize(
    "name",
    [
        "sd-v1-5-inpainting.ckpt",
        "sd-v1-5-inpainting.safetensors",
        "v1-5-pruned-emaonly.safetensors",
    ],
)
def test_local_file_path(sd_device, sampler, name):
    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    sd_steps = 30
    model = ModelManager(
        name=name,
        device=torch.device(sd_device),
        hf_access_token="",
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        cpu_offload=False,
    )
    cfg = get_config(
        HDStrategy.ORIGINAL,
        prompt="a fox sitting on a bench",
        sd_steps=sd_steps,
    )
    cfg.sd_sampler = sampler

    name = f"device_{sd_device}_{sampler}_{name}"

    assert_equal(
        model,
        cfg,
        f"sd_local_model_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )
