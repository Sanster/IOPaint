import io
from pathlib import Path
from typing import List

from PIL import Image

from lama_cleaner.helper import pil_to_bytes, load_img

current_dir = Path(__file__).parent.absolute().resolve()


def print_exif(exif):
    for k, v in exif.items():
        print(f"{k}: {v}")


def extra_info(img_p: Path):
    ext = img_p.suffix.strip(".")
    img_bytes = img_p.read_bytes()
    np_img, _, infos = load_img(img_bytes, False, True)
    pil_bytes = pil_to_bytes(Image.fromarray(np_img), ext=ext, infos=infos)
    res_img = Image.open(io.BytesIO(pil_bytes))
    return infos, res_img.info


def assert_keys(keys: List[str], infos, res_infos):
    for k in keys:
        assert k in infos
        assert k in res_infos
        assert infos[k] == res_infos[k]


def test_png_icc_profile_png():
    infos, res_infos = extra_info(current_dir / "icc_profile_test.png")
    assert_keys(["icc_profile", "exif"], infos, res_infos)


def test_png_icc_profile_jpeg():
    infos, res_infos = extra_info(current_dir / "icc_profile_test.jpg")
    assert_keys(["icc_profile", "exif"], infos, res_infos)


def test_jpeg():
    jpg_img_p = current_dir / "bunny.jpeg"
    infos, res_infos = extra_info(jpg_img_p)
    assert_keys(["dpi", "exif"], infos, res_infos)


def test_png_parameter():
    jpg_img_p = current_dir / "png_parameter_test.png"
    infos, res_infos = extra_info(jpg_img_p)
    assert_keys(["parameters"], infos, res_infos)
