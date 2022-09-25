import cv2
from lama_cleaner.model.base import InpaintModel
from lama_cleaner.schema import Config

class OpenCV2(InpaintModel):
    pad_mod = 1

    @staticmethod
    def is_downloaded() -> bool:
        return True

    def forward(self, image, mask, config: Config):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1]
        return: BGR IMAGE
        """
        cur_res = cv2.inpaint(image[:,:,::-1], mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return cur_res
