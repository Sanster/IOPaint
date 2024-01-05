import io
import tempfile
from pathlib import Path
from typing import List

from PIL import Image

from iopaint.helper import pil_to_bytes, load_img

current_dir = Path(__file__).parent.absolute().resolve()


def print_exif(exif):
    for k, v in exif.items():
        print(f"{k}: {v}")


def extra_info(img_p: Path):
    ext = img_p.suffix.strip(".")
    img_bytes = img_p.read_bytes()
    np_img, _, infos = load_img(img_bytes, False, True)
    res_pil_bytes = pil_to_bytes(Image.fromarray(np_img), ext=ext, infos=infos)
    res_img = Image.open(io.BytesIO(res_pil_bytes))
    return infos, res_img.info, res_pil_bytes


def assert_keys(keys: List[str], infos, res_infos):
    for k in keys:
        assert k in infos
        assert k in res_infos
        assert infos[k] == res_infos[k]


def run_test(file_path, keys):
    infos, res_infos, res_pil_bytes = extra_info(file_path)
    assert_keys(keys, infos, res_infos)
    with tempfile.NamedTemporaryFile("wb", suffix=file_path.suffix) as temp_file:
        temp_file.write(res_pil_bytes)
        temp_file.flush()
        infos, res_infos, res_pil_bytes = extra_info(Path(temp_file.name))
        assert_keys(keys, infos, res_infos)


def test_png_icc_profile_png():
    run_test(current_dir / "icc_profile_test.png", ["icc_profile", "exif"])


def test_png_icc_profile_jpeg():
    run_test(current_dir / "icc_profile_test.jpg", ["icc_profile", "exif"])


def test_jpeg():
    jpg_img_p = current_dir / "bunny.jpeg"
    run_test(jpg_img_p, ["dpi", "exif"])


def test_png_parameter():
    jpg_img_p = current_dir / "png_parameter_test.png"
    run_test(jpg_img_p, ["parameters"])
