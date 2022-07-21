import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler

current_dir = Path(__file__).parent.absolute().resolve()


def get_data(fx=1):
    img = cv2.imread(str(current_dir / "image.png"))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    mask = cv2.imread(str(current_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if fx != 1:
        img = cv2.resize(img, None, fx=fx, fy=1)
        mask = cv2.resize(mask, None, fx=fx, fy=1)
    return img, mask


def get_config(strategy, **kwargs):
    data = dict(
        ldm_steps=1,
        ldm_sampler=LDMSampler.plms,
        hd_strategy=strategy,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=200,
        hd_strategy_resize_limit=200,
    )
    data.update(**kwargs)
    return Config(**data)


def assert_equal(model, config, gt_name, fx=1):
    img, mask = get_data(fx=fx)
    res = model(img, mask, config)
    cv2.imwrite(
        str(current_dir / gt_name),
        res,
        [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(cv2.IMWRITE_PNG_COMPRESSION), 0],
    )

    """
    Note that JPEG is lossy compression, so even if it is the highest quality 100, 
    when the saved images is reloaded, a difference occurs with the original pixel value. 
    If you want to save the original images as it is, save it as PNG or BMP.
    """
    # gt = cv2.imread(str(current_dir / gt_name), cv2.IMREAD_UNCHANGED)
    # assert np.array_equal(res, gt)


@pytest.mark.parametrize(
    "strategy", [HDStrategy.ORIGINAL, HDStrategy.RESIZE, HDStrategy.CROP]
)
def test_lama(strategy):
    model = ModelManager(name="lama", device="cpu")
    assert_equal(
        model,
        get_config(strategy),
        f"lama_{strategy[0].upper() + strategy[1:]}_result.png",
    )

    fx = 1.3
    assert_equal(
        model,
        get_config(strategy),
        f"lama_{strategy[0].upper() + strategy[1:]}_fx_{fx}_result.png",
        fx=1.3,
    )


@pytest.mark.parametrize(
    "strategy", [HDStrategy.ORIGINAL, HDStrategy.RESIZE, HDStrategy.CROP]
)
@pytest.mark.parametrize("ldm_sampler", [LDMSampler.ddim, LDMSampler.plms])
def test_ldm(strategy, ldm_sampler):
    model = ModelManager(name="ldm", device="cpu")
    cfg = get_config(strategy, ldm_sampler=ldm_sampler)
    assert_equal(
        model, cfg, f"ldm_{strategy[0].upper() + strategy[1:]}_{ldm_sampler}_result.png"
    )

    fx = 1.3
    assert_equal(
        model,
        cfg,
        f"ldm_{strategy[0].upper() + strategy[1:]}_{ldm_sampler}_fx_{fx}_result.png",
        fx=fx,
    )


@pytest.mark.parametrize(
    "strategy", [HDStrategy.ORIGINAL, HDStrategy.RESIZE, HDStrategy.CROP]
)
@pytest.mark.parametrize("zits_wireframe", [False, True])
def test_zits(strategy, zits_wireframe):
    model = ModelManager(name="zits", device="cpu")
    cfg = get_config(strategy, zits_wireframe=zits_wireframe)
    # os.environ['ZITS_DEBUG_LINE_PATH'] = str(current_dir / 'zits_debug_line.jpg')
    # os.environ['ZITS_DEBUG_EDGE_PATH'] = str(current_dir / 'zits_debug_edge.jpg')
    assert_equal(
        model,
        cfg,
        f"zits_{strategy[0].upper() + strategy[1:]}_wireframe_{zits_wireframe}_result.png",
    )

    fx = 1.3
    assert_equal(
        model,
        cfg,
        f"zits_{strategy[0].upper() + strategy[1:]}_wireframe_{zits_wireframe}_fx_{fx}_result.png",
        fx=fx,
    )
