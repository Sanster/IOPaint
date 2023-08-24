from pathlib import Path

import pytest
import torch

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.tests.test_model import get_config, assert_equal
from lama_cleaner.schema import HDStrategy

current_dir = Path(__file__).parent.absolute().resolve()
save_dir = current_dir / 'result'
save_dir.mkdir(exist_ok=True, parents=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.mark.parametrize("disable_nsfw", [True, False])
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_instruct_pix2pix(disable_nsfw, cpu_offload):
    sd_steps = 50 if device == 'cuda' else 1
    model = ModelManager(name="instruct_pix2pix",
                         device=torch.device(device),
                         hf_access_token="",
                         sd_run_local=False,
                         disable_nsfw=disable_nsfw,
                         sd_cpu_textencoder=False,
                         cpu_offload=cpu_offload)
    cfg = get_config(strategy=HDStrategy.ORIGINAL, prompt='What if it were snowing?', p2p_steps=sd_steps, sd_scale=1.1)

    name = f"device_{device}_disnsfw_{disable_nsfw}_cpu_offload_{cpu_offload}"

    assert_equal(
        model,
        cfg,
        f"instruct_pix2pix_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fx=1.3
    )


@pytest.mark.parametrize("disable_nsfw", [False])
@pytest.mark.parametrize("cpu_offload", [False])
def test_instruct_pix2pix_snow(disable_nsfw, cpu_offload):
    sd_steps = 50 if device == 'cuda' else 1
    model = ModelManager(name="instruct_pix2pix",
                         device=torch.device(device),
                         hf_access_token="",
                         sd_run_local=False,
                         disable_nsfw=disable_nsfw,
                         sd_cpu_textencoder=False,
                         cpu_offload=cpu_offload)
    cfg = get_config(strategy=HDStrategy.ORIGINAL, prompt='What if it were snowing?', p2p_steps=sd_steps)

    name = f"snow"

    assert_equal(
        model,
        cfg,
        f"instruct_pix2pix_{name}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )
