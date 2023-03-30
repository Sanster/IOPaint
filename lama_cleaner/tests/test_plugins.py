from pathlib import Path

import cv2
import pytest
import torch.cuda

from lama_cleaner.plugins import (
    RemoveBG,
    RealESRGANUpscaler,
    GFPGANPlugin,
    RestoreFormerPlugin,
)

current_dir = Path(__file__).parent.absolute().resolve()
save_dir = current_dir / "result"
save_dir.mkdir(exist_ok=True, parents=True)
img_p = current_dir / "bunny.jpeg"
bgr_img = cv2.imread(str(img_p))
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


def _save(img, name):
    cv2.imwrite(str(save_dir / name), img)


def test_remove_bg():
    model = RemoveBG()
    res = model.forward(bgr_img)
    _save(res, "test_remove_bg.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_upscale(device):
    if device == "cuda" and not torch.cuda.is_available():
        return
    if device == "mps" and not torch.backends.mps.is_available():
        return

    model = RealESRGANUpscaler("realesr-general-x4v3", device)
    res = model.forward(bgr_img, 2)
    _save(res, f"test_upscale_x2_{device}.png")

    res = model.forward(bgr_img, 4)
    _save(res, f"test_upscale_x4_{device}.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_gfpgan(device):
    if device == "cuda" and not torch.cuda.is_available():
        return
    if device == "mps" and not torch.backends.mps.is_available():
        return
    model = GFPGANPlugin(device)
    res = model(rgb_img, None, None)
    _save(res, f"test_gfpgan_{device}.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_restoreformer(device):
    if device == "cuda" and not torch.cuda.is_available():
        return
    if device == "mps" and not torch.backends.mps.is_available():
        return
    model = RestoreFormerPlugin(device)
    res = model(rgb_img, None, None)
    _save(res, f"test_restoreformer_{device}.png")
