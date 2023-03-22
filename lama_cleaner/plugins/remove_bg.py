import cv2
import numpy as np


class RemoveBG:
    name = "RemoveBG"

    def __init__(self):
        from rembg import new_session

        self.session = new_session(model_name="u2net")

    def __call__(self, rgb_np_img, files, form):
        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)
        return self.forward(bgr_np_img)

    def forward(self, bgr_np_img) -> np.ndarray:
        from rembg import remove

        # return BGRA image
        output = remove(bgr_np_img, session=self.session)
        return output
