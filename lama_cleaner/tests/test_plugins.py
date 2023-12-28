import hashlib
import os
import time

from lama_cleaner.plugins.anime_seg import AnimeSeg
from lama_cleaner.tests.utils import check_device, current_dir, save_dir

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import pytest

from lama_cleaner.plugins import (
    RemoveBG,
    RealESRGANUpscaler,
    GFPGANPlugin,
    RestoreFormerPlugin,
    InteractiveSeg,
)

img_p = current_dir / "bunny.jpeg"
img_bytes = open(img_p, "rb").read()
bgr_img = cv2.imread(str(img_p))
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


def _save(img, name):
    cv2.imwrite(str(save_dir / name), img)


def test_remove_bg():
    model = RemoveBG()
    res = model.forward(bgr_img)
    res = cv2.cvtColor(res, cv2.COLOR_RGBA2BGRA)
    _save(res, "test_remove_bg.png")


def test_anime_seg():
    model = AnimeSeg()
    img = cv2.imread(str(current_dir / "anime_test.png"))
    res = model.forward(img)
    assert len(res.shape) == 3
    assert res.shape[-1] == 4
    _save(res, "test_anime_seg.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_upscale(device):
    check_device(device)
    model = RealESRGANUpscaler("realesr-general-x4v3", device)
    res = model.forward(bgr_img, 2)
    _save(res, f"test_upscale_x2_{device}.png")

    res = model.forward(bgr_img, 4)
    _save(res, f"test_upscale_x4_{device}.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_gfpgan(device):
    check_device(device)
    model = GFPGANPlugin(device)
    res = model(rgb_img, None, None)
    _save(res, f"test_gfpgan_{device}.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_restoreformer(device):
    check_device(device)
    model = RestoreFormerPlugin(device)
    res = model(rgb_img, None, None)
    _save(res, f"test_restoreformer_{device}.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_segment_anything(device):
    check_device(device)
    img_md5 = hashlib.md5(img_bytes).hexdigest()
    model = InteractiveSeg("vit_l", device)
    new_mask = model.forward(rgb_img, [[448 // 2, 394 // 2, 1]], img_md5)

    save_name = f"test_segment_anything_{device}.png"
    _save(new_mask, save_name)

    start = time.time()
    model.forward(rgb_img, [[448 // 2, 394 // 2, 1]], img_md5)
    print(f"Time for {save_name}: {time.time() - start:.2f}s")
