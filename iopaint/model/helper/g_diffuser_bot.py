import cv2
import numpy as np


def expand_image(cv2_img, top: int, right: int, bottom: int, left: int):
    assert cv2_img.shape[2] == 3
    origin_h, origin_w = cv2_img.shape[:2]

    # TODO: which is better?
    # new_img = np.ones((new_height, new_width, 3), np.uint8) * 255
    new_img = cv2.copyMakeBorder(
        cv2_img, top, bottom, left, right, cv2.BORDER_REPLICATE
    )

    inner_padding_left = 0 if left > 0 else 0
    inner_padding_right = 0 if right > 0 else 0
    inner_padding_top = 0 if top > 0 else 0
    inner_padding_bottom = 0 if bottom > 0 else 0

    mask_image = np.zeros(
        (
            origin_h - inner_padding_top - inner_padding_bottom,
            origin_w - inner_padding_left - inner_padding_right,
        ),
        np.uint8,
    )
    mask_image = cv2.copyMakeBorder(
        mask_image,
        top + inner_padding_top,
        bottom + inner_padding_bottom,
        left + inner_padding_left,
        right + inner_padding_right,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    # k = 2*int(min(origin_h, origin_w) // 6)+1
    # k = 7
    # mask_image = cv2.GaussianBlur(mask_image, (k, k), 0)
    return new_img, mask_image


if __name__ == "__main__":
    from pathlib import Path

    current_dir = Path(__file__).parent.absolute().resolve()
    image_path = "/Users/cwq/code/github/IOPaint/iopaint/tests/bunny.jpeg"
    init_image = cv2.imread(str(image_path))
    init_image, mask_image = expand_image(
        init_image,
        top=0,
        right=0,
        bottom=0,
        left=100,
        softness=20,
        space=20,
    )
    print(mask_image.dtype, mask_image.min(), mask_image.max())
    print(init_image.dtype, init_image.min(), init_image.max())
    mask_image = mask_image.astype(np.uint8)
    init_image = init_image.astype(np.uint8)
    cv2.imwrite("expanded_image.png", init_image)
    cv2.imwrite("expanded_mask.png", mask_image)
