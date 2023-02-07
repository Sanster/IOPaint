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


@pytest.mark.parametrize("sd_device", ["cuda"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
@pytest.mark.parametrize("cpu_textencoder", [True, False])
@pytest.mark.parametrize("disable_nsfw", [True, False])
def test_runway_sd_1_5_ddim(
    sd_device, strategy, sampler, cpu_textencoder, disable_nsfw
):
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
        disable_nsfw=disable_nsfw,
        sd_cpu_textencoder=cpu_textencoder,
        callback=callback,
    )
    cfg = get_config(strategy, prompt="a fox sitting on a bench", sd_steps=sd_steps)
    cfg.sd_sampler = sampler

    name = f"device_{sd_device}_{sampler}_cpu_textencoder_{cpu_textencoder}_disnsfw_{disable_nsfw}"

    assert_equal(
        model,
        cfg,
        f"runway_sd_{strategy.capitalize()}_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1.3,
    )


@pytest.mark.parametrize("sd_device", ["cuda"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize(
    "sampler", [SDSampler.pndm, SDSampler.k_lms, SDSampler.k_euler, SDSampler.k_euler_a]
)
@pytest.mark.parametrize("cpu_textencoder", [False])
@pytest.mark.parametrize("disable_nsfw", [True])
def test_runway_sd_1_5(sd_device, strategy, sampler, cpu_textencoder, disable_nsfw):
    def callback(i, t, latents):
        print(f"sd_step_{i}")

    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    sd_steps = 50 if sd_device == "cuda" else 1
    model = ModelManager(
        name="sd1.5",
        device=torch.device(sd_device),
        hf_access_token="",
        sd_run_local=True,
        disable_nsfw=disable_nsfw,
        sd_cpu_textencoder=cpu_textencoder,
        callback=callback,
    )
    cfg = get_config(strategy, prompt="a fox sitting on a bench", sd_steps=sd_steps)
    cfg.sd_sampler = sampler

    name = f"device_{sd_device}_{sampler}_cpu_textencoder_{cpu_textencoder}_disnsfw_{disable_nsfw}"

    assert_equal(
        model,
        cfg,
        f"runway_sd_{strategy.capitalize()}_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1.3,
    )


@pytest.mark.parametrize("sd_device", ["cuda"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.ddim])
def test_runway_sd_1_5_negative_prompt(sd_device, strategy, sampler):
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
        disable_nsfw=False,
        sd_cpu_textencoder=False,
        callback=callback,
    )
    cfg = get_config(
        strategy,
        sd_steps=sd_steps,
        prompt="Face of a fox, high resolution, sitting on a park bench",
        negative_prompt="orange, yellow, small",
        sd_sampler=sampler,
        sd_match_histograms=True,
    )

    name = f"{sampler}_negative_prompt"

    assert_equal(
        model,
        cfg,
        f"runway_sd_{strategy.capitalize()}_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1,
    )


@pytest.mark.parametrize("sd_device", ["cuda"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.k_euler_a])
@pytest.mark.parametrize("cpu_textencoder", [False])
@pytest.mark.parametrize("disable_nsfw", [False])
def test_runway_sd_1_5_sd_scale(
    sd_device, strategy, sampler, cpu_textencoder, disable_nsfw
):
    if sd_device == "cuda" and not torch.cuda.is_available():
        return

    sd_steps = 50 if sd_device == "cuda" else 1
    model = ModelManager(
        name="sd1.5",
        device=torch.device(sd_device),
        hf_access_token="",
        sd_run_local=True,
        disable_nsfw=disable_nsfw,
        sd_cpu_textencoder=cpu_textencoder,
    )
    cfg = get_config(
        strategy, prompt="a fox sitting on a bench", sd_steps=sd_steps, sd_scale=0.85
    )
    cfg.sd_sampler = sampler

    name = f"device_{sd_device}_{sampler}_cpu_textencoder_{cpu_textencoder}_disnsfw_{disable_nsfw}"

    assert_equal(
        model,
        cfg,
        f"runway_sd_{strategy.capitalize()}_{name}_sdscale.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1.3,
    )


@pytest.mark.parametrize("sd_device", ["cuda"])
@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
@pytest.mark.parametrize("sampler", [SDSampler.k_euler_a])
def test_runway_sd_1_5_cpu_offload(sd_device, strategy, sampler):
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
        cpu_offload=True,
    )
    cfg = get_config(
        strategy, prompt="a fox sitting on a bench", sd_steps=sd_steps, sd_scale=0.85
    )
    cfg.sd_sampler = sampler

    name = f"device_{sd_device}_{sampler}"

    assert_equal(
        model,
        cfg,
        f"runway_sd_{strategy.capitalize()}_{name}_cpu_offload.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )

