from pathlib import Path

import cv2
import pytest
import torch
from PIL import Image

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import HDStrategy
from lama_cleaner.tests.test_model import get_config, get_data

current_dir = Path(__file__).parent.absolute().resolve()
save_dir = current_dir / 'result'
save_dir.mkdir(exist_ok=True, parents=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


def assert_equal(
    model, config, gt_name,
    fx: float = 1, fy: float = 1,
    img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
    mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    example_p=current_dir / "bunny.jpeg",
):
    img, mask = get_data(fx=fx, fy=fy, img_p=img_p, mask_p=mask_p)

    example_image = cv2.imread(str(example_p))
    example_image = cv2.cvtColor(example_image, cv2.COLOR_BGRA2RGB)
    example_image = cv2.resize(example_image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

    print(f"Input image shape: {img.shape}, example_image: {example_image.shape}")
    config.paint_by_example_example_image = Image.fromarray(example_image)
    res = model(img, mask, config)
    cv2.imwrite(str(save_dir / gt_name), res)


@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
def test_paint_by_example(strategy):
    model = ModelManager(name="paint_by_example", device=device, disable_nsfw=True)
    cfg = get_config(strategy, paint_by_example_steps=30)
    assert_equal(
        model,
        cfg,
        f"paint_by_example_{strategy.capitalize()}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fy=0.9,
        fx=1.3,
    )


@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
def test_paint_by_example_disable_nsfw(strategy):
    model = ModelManager(name="paint_by_example", device=device, disable_nsfw=False)
    cfg = get_config(strategy, paint_by_example_steps=30)
    assert_equal(
        model,
        cfg,
        f"paint_by_example_{strategy.capitalize()}_disable_nsfw.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
def test_paint_by_example_sd_scale(strategy):
    model = ModelManager(name="paint_by_example", device=device, disable_nsfw=True)
    cfg = get_config(strategy, paint_by_example_steps=30, sd_scale=0.85)
    assert_equal(
        model,
        cfg,
        f"paint_by_example_{strategy.capitalize()}_sdscale.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fy=0.9,
        fx=1.3
    )


@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
def test_paint_by_example_cpu_offload(strategy):
    model = ModelManager(name="paint_by_example", device=device, cpu_offload=True, disable_nsfw=False)
    cfg = get_config(strategy, paint_by_example_steps=30, sd_scale=0.85)
    assert_equal(
        model,
        cfg,
        f"paint_by_example_{strategy.capitalize()}_cpu_offload.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    )


@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
def test_paint_by_example_cpu_offload_cpu_device(strategy):
    model = ModelManager(name="paint_by_example", device=torch.device('cpu'), cpu_offload=True, disable_nsfw=True)
    cfg = get_config(strategy, paint_by_example_steps=1, sd_scale=0.85)
    assert_equal(
        model,
        cfg,
        f"paint_by_example_{strategy.capitalize()}_cpu_offload_cpu_device.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fy=0.9,
        fx=1.3
    )
