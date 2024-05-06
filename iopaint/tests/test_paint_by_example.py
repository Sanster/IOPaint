import cv2
import pytest
from PIL import Image
from iopaint.helper import encode_pil_to_base64

from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy
from iopaint.tests.utils import (
    current_dir,
    get_config,
    get_data,
    save_dir,
    check_device,
)

model_name = "Fantasy-Studio/Paint-by-Example"


def assert_equal(
    model,
    config,
    save_name: str,
    fx: float = 1,
    fy: float = 1,
    img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
    mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
    example_p=current_dir / "bunny.jpeg",
):
    img, mask = get_data(fx=fx, fy=fy, img_p=img_p, mask_p=mask_p)

    example_image = cv2.imread(str(example_p))
    example_image = cv2.cvtColor(example_image, cv2.COLOR_BGRA2RGB)
    example_image = cv2.resize(
        example_image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA
    )

    print(f"Input image shape: {img.shape}, example_image: {example_image.shape}")
    config.paint_by_example_example_image = encode_pil_to_base64(
        Image.fromarray(example_image), 100, {}
    ).decode("utf-8")
    res = model(img, mask, config)
    cv2.imwrite(str(save_dir / save_name), res)


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_paint_by_example(device):
    sd_steps = check_device(device)
    model = ModelManager(name=model_name, device=device, disable_nsfw=True)
    cfg = get_config(strategy=HDStrategy.ORIGINAL, sd_steps=sd_steps)
    assert_equal(
        model,
        cfg,
        f"paint_by_example_device_{device}.png",
        img_p=current_dir / "overture-creations-5sI6fQgYIuo.png",
        mask_p=current_dir / "overture-creations-5sI6fQgYIuo_mask.png",
        fy=0.9,
        fx=1.3,
    )
