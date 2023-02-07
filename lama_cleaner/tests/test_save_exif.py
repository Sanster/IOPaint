import io
from pathlib import Path

from PIL import Image

from lama_cleaner.helper import pil_to_bytes


current_dir = Path(__file__).parent.absolute().resolve()
png_img_p = current_dir / "image.png"
jpg_img_p = current_dir / "bunny.jpeg"


def print_exif(exif):
    for k, v in exif.items():
        print(f"{k}: {v}")


def test_png():
    img = Image.open(png_img_p)
    exif = img.getexif()
    print_exif(exif)

    pil_bytes = pil_to_bytes(img, ext="png", exif=exif)

    res_img = Image.open(io.BytesIO(pil_bytes))
    res_exif = res_img.getexif()

    assert dict(exif) == dict(res_exif)


def test_jpeg():
    img = Image.open(jpg_img_p)
    exif = img.getexif()
    print_exif(exif)

    pil_bytes = pil_to_bytes(img, ext="jpeg", exif=exif)

    res_img = Image.open(io.BytesIO(pil_bytes))
    res_exif = res_img.getexif()

    assert dict(exif) == dict(res_exif)
