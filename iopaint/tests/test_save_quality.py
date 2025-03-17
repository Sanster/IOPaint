import os
import io
from PIL import Image
from iopaint.helper import pil_to_bytes

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_jpeg_quality():
    # Test JPEG quality settings
    img_path = os.path.join(TESTS_DIR, "bunny.jpeg")
    pil_img = Image.open(img_path)

    # Test different quality settings
    high_quality = pil_to_bytes(pil_img, "jpg", quality=95)
    low_quality = pil_to_bytes(pil_img, "jpg", quality=50)

    # Print file sizes in KB
    print(f"High quality JPEG size: {len(high_quality) / 1024:.2f} KB")
    print(f"Low quality JPEG size: {len(low_quality) / 1024:.2f} KB")

    # Higher quality should result in larger file size
    assert len(high_quality) > len(low_quality)

    # Verify the output can be opened as an image
    Image.open(io.BytesIO(high_quality))
    Image.open(io.BytesIO(low_quality))


def test_png_parameters():
    # Test PNG with parameters
    img_path = os.path.join(TESTS_DIR, "cat.png")
    pil_img = Image.open(img_path)

    # Test PNG with parameters
    params = {"parameters": "test_param=value"}
    png_with_params = pil_to_bytes(pil_img, "png", infos=params)

    # Test PNG without parameters
    png_without_params = pil_to_bytes(pil_img, "png")

    # Print file sizes in KB
    print(f"PNG with parameters size: {len(png_with_params) / 1024:.2f} KB")
    print(f"PNG without parameters size: {len(png_without_params) / 1024:.2f} KB")

    # Verify both outputs can be opened as images
    Image.open(io.BytesIO(png_with_params))
    Image.open(io.BytesIO(png_without_params))


def test_format_conversion():
    # Test format conversion
    jpeg_path = os.path.join(TESTS_DIR, "bunny.jpeg")
    png_path = os.path.join(TESTS_DIR, "cat.png")

    # Convert JPEG to PNG
    jpeg_img = Image.open(jpeg_path)
    jpeg_to_png = pil_to_bytes(jpeg_img, "png")
    converted_png = Image.open(io.BytesIO(jpeg_to_png))
    print(f"JPEG to PNG size: {len(jpeg_to_png) / 1024:.2f} KB")
    assert converted_png.format.lower() == "png"

    # Convert PNG to JPEG
    png_img = Image.open(png_path)
    # Convert RGBA to RGB if necessary
    if png_img.mode == "RGBA":
        png_img = png_img.convert("RGB")
    png_to_jpeg = pil_to_bytes(png_img, "jpg")
    print(f"PNG to JPEG size: {len(png_to_jpeg) / 1024:.2f} KB")
    converted_jpeg = Image.open(io.BytesIO(png_to_jpeg))
    assert converted_jpeg.format.lower() == "jpeg"
