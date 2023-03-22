from pathlib import Path

import cv2

from lama_cleaner.plugins import RemoveBG, RealESRGANUpscaler

current_dir = Path(__file__).parent.absolute().resolve()
save_dir = current_dir / "result"
save_dir.mkdir(exist_ok=True, parents=True)
img_p = current_dir / "bunny.jpeg"


def test_remove_bg():
    model = RemoveBG()
    img = cv2.imread(str(img_p))
    res = model.forward(img)
    cv2.imwrite(str(save_dir / "test_remove_bg.png"), res)


def test_upscale():
    model = RealESRGANUpscaler("cpu")
    img = cv2.imread(str(img_p))
    res = model.forward(img, 2)
    cv2.imwrite(str(save_dir / "test_upscale_x2.png"), res)

    res = model.forward(img, 4)
    cv2.imwrite(str(save_dir / "test_upscale_x4.png"), res)
