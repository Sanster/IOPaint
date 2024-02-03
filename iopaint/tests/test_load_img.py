from iopaint.helper import load_img
from iopaint.tests.utils import current_dir

png_img_p = current_dir / "image.png"
jpg_img_p = current_dir / "bunny.jpeg"


def test_load_png_image():
    with open(png_img_p, "rb") as f:
        np_img, alpha_channel = load_img(f.read())
    assert np_img.shape == (256, 256, 3)
    assert alpha_channel.shape == (256, 256)


def test_load_jpg_image():
    with open(jpg_img_p, "rb") as f:
        np_img, alpha_channel = load_img(f.read())
    assert np_img.shape == (394, 448, 3)
    assert alpha_channel is None
