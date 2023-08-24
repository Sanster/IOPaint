import io
from pathlib import Path

from PIL import Image

from lama_cleaner.helper import pil_to_bytes, load_img

current_dir = Path(__file__).parent.absolute().resolve()


def print_exif(exif):
    for k, v in exif.items():
        print(f"{k}: {v}")


def run_test(img_p: Path):
    print(img_p)
    ext = img_p.suffix.strip(".")
    img_bytes = img_p.read_bytes()
    np_img, _, exif_infos = load_img(img_bytes, False, True)
    print(exif_infos)
    print("Original exif_infos")
    print_exif(exif_infos["exif"])

    pil_to_bytes(Image.fromarray(np_img), ext=ext, exif_infos={})

    pil_bytes = pil_to_bytes(Image.fromarray(np_img), ext=ext, exif_infos=exif_infos)
    res_img = Image.open(io.BytesIO(pil_bytes))
    print(f"Result img info: {res_img.info}")
    res_exif = res_img.getexif()
    print_exif(res_exif)
    assert res_exif == exif_infos["exif"]
    assert exif_infos["parameters"] == res_img.info.get("parameters")


def test_png():
    run_test(current_dir / "image.png")
    run_test(current_dir / "pnginfo_test.png")


def test_jpeg():
    jpg_img_p = current_dir / "bunny.jpeg"
    run_test(jpg_img_p)
