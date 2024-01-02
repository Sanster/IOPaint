import hashlib
import os
import time
from PIL import Image

from lama_cleaner.helper import encode_pil_to_base64
from lama_cleaner.plugins.anime_seg import AnimeSeg
from lama_cleaner.schema import RunPluginRequest
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
rgb_img_base64 = encode_pil_to_base64(Image.fromarray(rgb_img), 100, {})
bgr_img_base64 = encode_pil_to_base64(Image.fromarray(bgr_img), 100, {})


def _save(img, name):
    cv2.imwrite(str(save_dir / name), img)


def test_remove_bg():
    model = RemoveBG()
    rgba_np_img = model(
        rgb_img, RunPluginRequest(name=RemoveBG.name, image=rgb_img_base64)
    )
    res = cv2.cvtColor(rgba_np_img, cv2.COLOR_RGBA2BGRA)
    _save(res, "test_remove_bg.png")


def test_anime_seg():
    model = AnimeSeg()
    img = cv2.imread(str(current_dir / "anime_test.png"))
    img_base64 = encode_pil_to_base64(Image.fromarray(img), 100, {})
    res = model(img, RunPluginRequest(name=AnimeSeg.name, image=img_base64))
    assert len(res.shape) == 3
    assert res.shape[-1] == 4
    _save(res, "test_anime_seg.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_upscale(device):
    check_device(device)
    model = RealESRGANUpscaler("realesr-general-x4v3", device)
    res = model(
        rgb_img,
        RunPluginRequest(name=RealESRGANUpscaler.name, image=rgb_img_base64, scale=2),
    )
    _save(res, f"test_upscale_x2_{device}.png")

    res = model(
        rgb_img,
        RunPluginRequest(name=RealESRGANUpscaler.name, image=rgb_img_base64, scale=4),
    )
    _save(res, f"test_upscale_x4_{device}.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_gfpgan(device):
    check_device(device)
    model = GFPGANPlugin(device)
    res = model(rgb_img, RunPluginRequest(name=GFPGANPlugin.name, image=rgb_img_base64))
    _save(res, f"test_gfpgan_{device}.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_restoreformer(device):
    check_device(device)
    model = RestoreFormerPlugin(device)
    res = model(
        rgb_img, RunPluginRequest(name=RestoreFormerPlugin.name, image=rgb_img_base64)
    )
    _save(res, f"test_restoreformer_{device}.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_segment_anything(device):
    check_device(device)
    model = InteractiveSeg("vit_l", device)
    new_mask = model(
        rgb_img,
        RunPluginRequest(
            name=InteractiveSeg.name,
            image=rgb_img_base64,
            clicks=([[448 // 2, 394 // 2, 1]]),
        ),
    )

    save_name = f"test_segment_anything_{device}.png"
    _save(new_mask, save_name)
