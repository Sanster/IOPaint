import os
import cv2
import numpy as np
from torch.hub import get_dir

from lama_cleaner.plugins.base_plugin import BasePlugin


class RemoveBG(BasePlugin):
    name = "RemoveBG"

    def __init__(self):
        super().__init__()
        from rembg import new_session

        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")
        os.environ["U2NET_HOME"] = model_dir

        self.session = new_session(model_name="u2net")

    def __call__(self, rgb_np_img, files, form):
        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)
        return self.forward(bgr_np_img)

    def forward(self, bgr_np_img) -> np.ndarray:
        from rembg import remove

        # return BGRA image
        output = remove(bgr_np_img, session=self.session)
        return cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)

    def check_dep(self):
        try:
            import rembg
        except ImportError:
            return (
                "RemoveBG is not installed, please install it first. pip install rembg"
            )
