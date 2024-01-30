import cv2
from iopaint.helper import adjust_mask
from iopaint.tests.utils import current_dir, save_dir

mask_p = current_dir / "overture-creations-5sI6fQgYIuo_mask.png"


def test_adjust_mask():
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    res_mask = adjust_mask(mask, 0, "expand")
    cv2.imwrite(str(save_dir / "adjust_mask_original.png"), res_mask)
    res_mask = adjust_mask(mask, 40, "expand")
    cv2.imwrite(str(save_dir / "adjust_mask_expand.png"), res_mask)
    res_mask = adjust_mask(mask, 20, "shrink")
    cv2.imwrite(str(save_dir / "adjust_mask_shrink.png"), res_mask)
    res_mask = adjust_mask(mask, 20, "reverse")
    cv2.imwrite(str(save_dir / "adjust_mask_reverse.png"), res_mask)
