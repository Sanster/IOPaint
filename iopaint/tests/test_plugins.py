import os
from PIL import Image

from iopaint.helper import encode_pil_to_base64, gen_frontend_mask
from iopaint.plugins.anime_seg import AnimeSeg
from iopaint.schema import Device, RunPluginRequest, RemoveBGModel, InteractiveSegModel
from iopaint.tests.utils import check_device, current_dir, save_dir

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import pytest

from iopaint.plugins import (
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

person_p = current_dir / "image.png"
person_bgr_img = cv2.imread(str(person_p))
person_rgb_img = cv2.cvtColor(person_bgr_img, cv2.COLOR_BGR2RGB)
person_rgb_img = cv2.resize(person_rgb_img, (512, 512))


def _save(img, name):
    name = name.replace("/", "_")
    cv2.imwrite(str(save_dir / name), img)


@pytest.mark.parametrize("model_name", RemoveBGModel.values())
@pytest.mark.parametrize("device", Device.values())
def test_remove_bg(model_name, device):
    check_device(device)
    print(f"Testing {model_name} on {device}")
    model = RemoveBG(model_name, device)
    rgba_np_img = model.gen_image(
        rgb_img, RunPluginRequest(name=RemoveBG.name, image=rgb_img_base64)
    )
    res = cv2.cvtColor(rgba_np_img, cv2.COLOR_RGBA2BGRA)
    _save(res, f"test_remove_bg_{model_name}_{device}.png")

    bgr_np_img = model.gen_mask(
        rgb_img, RunPluginRequest(name=RemoveBG.name, image=rgb_img_base64)
    )

    res_mask = gen_frontend_mask(bgr_np_img)
    _save(res_mask, f"test_remove_bg_frontend_mask_{model_name}_{device}.png")

    assert len(bgr_np_img.shape) == 2
    _save(bgr_np_img, f"test_remove_bg_mask_{model_name}_{device}.jpeg")


def test_anime_seg():
    model = AnimeSeg()
    img = cv2.imread(str(current_dir / "anime_test.png"))
    img_base64 = encode_pil_to_base64(Image.fromarray(img), 100, {})
    res = model.gen_image(img, RunPluginRequest(name=AnimeSeg.name, image=img_base64))
    assert len(res.shape) == 3
    assert res.shape[-1] == 4
    _save(res, "test_anime_seg.png")

    res = model.gen_mask(img, RunPluginRequest(name=AnimeSeg.name, image=img_base64))
    assert len(res.shape) == 2
    _save(res, "test_anime_seg_mask.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_upscale(device):
    check_device(device)
    model = RealESRGANUpscaler("realesr-general-x4v3", device)
    res = model.gen_image(
        rgb_img,
        RunPluginRequest(name=RealESRGANUpscaler.name, image=rgb_img_base64, scale=2),
    )
    _save(res, f"test_upscale_x2_{device}.png")

    res = model.gen_image(
        rgb_img,
        RunPluginRequest(name=RealESRGANUpscaler.name, image=rgb_img_base64, scale=4),
    )
    _save(res, f"test_upscale_x4_{device}.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_gfpgan(device):
    check_device(device)
    model = GFPGANPlugin(device)
    res = model.gen_image(
        person_rgb_img, RunPluginRequest(name=GFPGANPlugin.name, image=rgb_img_base64)
    )
    _save(res, f"test_gfpgan_{device}.png")


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_restoreformer(device):
    check_device(device)
    model = RestoreFormerPlugin(device)
    res = model.gen_image(
        person_rgb_img,
        RunPluginRequest(name=RestoreFormerPlugin.name, image=rgb_img_base64),
    )
    _save(res, f"test_restoreformer_{device}.png")


@pytest.mark.parametrize("name", InteractiveSegModel.values())
@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_segment_anything(name, device):
    check_device(device)
    model = InteractiveSeg(name, device)
    new_mask = model.gen_mask(
        rgb_img,
        RunPluginRequest(
            name=InteractiveSeg.name,
            image=rgb_img_base64,
            clicks=([[448 // 2, 394 // 2, 1]]),
        ),
    )

    save_name = f"test_segment_anything_{name}_{device}.png"
    _save(new_mask, save_name)
