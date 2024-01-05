import cv2
from .base import InpaintModel
from iopaint.schema import InpaintRequest

flag_map = {"INPAINT_NS": cv2.INPAINT_NS, "INPAINT_TELEA": cv2.INPAINT_TELEA}


class OpenCV2(InpaintModel):
    name = "cv2"
    pad_mod = 1
    is_erase_model = True

    @staticmethod
    def is_downloaded() -> bool:
        return True

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1]
        return: BGR IMAGE
        """
        cur_res = cv2.inpaint(
            image[:, :, ::-1],
            mask,
            inpaintRadius=config.cv2_radius,
            flags=flag_map[config.cv2_flag],
        )
        return cur_res
