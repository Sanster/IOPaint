import os
from pathlib import Path

import cv2
import pytest
import torch

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler

current_dir = Path(__file__).parent.absolute().resolve()
save_dir = current_dir / "result"
save_dir.mkdir(exist_ok=True, parents=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


def get_data(
    fx: float = 1,
    fy: float = 1.0,
    img_p=current_dir / "image.png",
    mask_p=current_dir / "mask.png",
):
    img = cv2.imread(str(img_p))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
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


def assert_equal(
    model,
    config,
    gt_name,
    fx: float = 1,
    fy: float = 1,
    img_p=current_dir / "image.png",
    mask_p=current_dir / "mask.png",
):
    img, mask = get_data(fx=fx, fy=fy, img_p=img_p, mask_p=mask_p)
    print(f"Input image shape: {img.shape}")
    res = model(img, mask, config)
    cv2.imwrite(
        str(save_dir / gt_name),
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


@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL])
def test_mat(strategy):
    model = ModelManager(name="mat", device=device)
    cfg = get_config(strategy)

    for _ in range(10):
        assert_equal(
            model, cfg, f"mat_{strategy.capitalize()}_result.png",
        )

