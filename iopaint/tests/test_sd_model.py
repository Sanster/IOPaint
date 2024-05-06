import os

from loguru import logger

from iopaint.tests.utils import check_device, get_config, assert_equal

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import pytest
import torch

from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, SDSampler

current_dir = Path(__file__).parent.absolute().resolve()
save_dir = current_dir / "result"
save_dir.mkdir(exist_ok=True, parents=True)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_runway_sd_1_5_all_samplers(device):
    sd_steps = check_device(device)
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )

    all_samplers = [member.value for member in SDSampler.__members__.values()]
    print(all_samplers)
    for sampler in all_samplers:
        print(f"Testing sampler {sampler}")
        if (
            sampler
            in [SDSampler.dpm2_karras, SDSampler.dpm2_a_karras, SDSampler.lms_karras]
            and device == "mps"
        ):
            # diffusers 0.25.0 still has bug on these sampler on mps, wait main branch released to fix it
            logger.warning(
                "skip dpm2_karras on mps, diffusers does not support it on mps. TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead."
            )
            continue
        cfg = get_config(
            strategy=HDStrategy.ORIGINAL,
            prompt="a fox sitting on a bench",
            sd_steps=sd_steps,
            sd_sampler=sampler,
        )

        name = f"device_{device}_{sampler}"

        assert_equal(
            model,
            cfg,
            f"runway_sd_{name}.png",
            img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
            mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        )


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize("sampler", [SDSampler.lcm])
def test_runway_sd_lcm_lora(device, sampler):
    check_device(device)

    sd_steps = 5
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )
    cfg = get_config(
        strategy=HDStrategy.ORIGINAL,
        prompt="face of a fox, sitting on a bench",
        sd_steps=sd_steps,
        sd_guidance_scale=2,
        sd_lcm_lora=True,
    )
    cfg.sd_sampler = sampler

    assert_equal(
        model,
        cfg,
        f"runway_sd_1_5_lcm_lora_device_{device}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
def test_runway_sd_sd_strength(device, strategy, sampler):
    sd_steps = check_device(device)
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )
    cfg = get_config(
        strategy=strategy,
        prompt="a fox sitting on a bench",
        sd_steps=sd_steps,
        sd_strength=0.8,
    )
    cfg.sd_sampler = sampler

    assert_equal(
        model,
        cfg,
        f"runway_sd_strength_0.8_device_{device}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
def test_runway_sd_cpu_textencoder(device, strategy, sampler):
    sd_steps = check_device(device)
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=True,
    )
    cfg = get_config(
        strategy=strategy,
        prompt="a fox sitting on a bench",
        sd_steps=sd_steps,
        sd_sampler=sampler,
    )

    assert_equal(
        model,
        cfg,
        f"runway_sd_device_{device}_cpu_textencoder.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
def test_runway_norm_sd_model(device, strategy, sampler):
    sd_steps = check_device(device)
    model = ModelManager(
        name="runwayml/stable-diffusion-v1-5",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
    )
    cfg = get_config(
        strategy=strategy, prompt="face of a fox, sitting on a bench", sd_steps=sd_steps
    )
    cfg.sd_sampler = sampler

    assert_equal(
        model,
        cfg,
        f"runway_{device}_norm_sd_model_device_{device}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.dpm_plus_plus_2m])
def test_runway_sd_1_5_cpu_offload(device, strategy, sampler):
    sd_steps = check_device(device)
    model = ModelManager(
        name="runwayml/stable-diffusion-inpainting",
        device=torch.device(device),
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        cpu_offload=True,
    )
    cfg = get_config(
        strategy=strategy, prompt="a fox sitting on a bench", sd_steps=sd_steps
    )
    cfg.sd_sampler = sampler

    name = f"device_{device}_{sampler}"

    assert_equal(
        model,
        cfg,
        f"runway_sd_{strategy.capitalize()}_{name}_cpu_offload.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
@pytest.mark.parametrize(
    "name",
    [
        "sd-v1-5-inpainting.safetensors",
        "v1-5-pruned-emaonly.safetensors",
        "sd_xl_base_1.0.safetensors",
        "sd_xl_base_1.0_inpainting_0.1.safetensors",
    ],
)
def test_local_file_path(device, sampler, name):
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
        prompt="a fox sitting on a bench",
        sd_steps=sd_steps,
    )
    cfg.sd_sampler = sampler

    name = f"device_{device}_{sampler}_{name}"

    is_sdxl = "sd_xl" in name

    assert_equal(
        model,
        cfg,
        f"sd_local_model_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1.5 if is_sdxl else 1,
        fy=1.5 if is_sdxl else 1,
    )
