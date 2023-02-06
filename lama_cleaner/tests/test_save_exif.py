import io

from PIL import Image

from lama_cleaner.helper import pil_to_bytes


def print_exif(exif):
    for k, v in exif.items():
        print(f"{k}: {v}")


def test_png():
    img = Image.open("image.png")
    exif = img.getexif()
    print_exif(exif)

    pil_bytes = pil_to_bytes(img, ext="png", exif=exif)

    res_img = Image.open(io.BytesIO(pil_bytes))
    res_exif = res_img.getexif()

    assert dict(exif) == dict(res_exif)


def test_jpeg():
    img = Image.open("bunny.jpeg")
    exif = img.getexif()
    print_exif(exif)

    pil_bytes = pil_to_bytes(img, ext="jpeg", exif=exif)

    res_img = Image.open(io.BytesIO(pil_bytes))
    res_exif = res_img.getexif()

    assert dict(exif) == dict(res_exif)
