import os
import cv2
import numpy as np
from torch.hub import get_dir

from iopaint.plugins.base_plugin import BasePlugin
from iopaint.schema import RunPluginRequest


class RemoveBG(BasePlugin):
    name = "RemoveBG"
    support_gen_mask = True
    support_gen_image = True

    def __init__(self):
        super().__init__()
        from rembg import new_session

        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")
        os.environ["U2NET_HOME"] = model_dir

        self.session = new_session(model_name="u2net")

    def gen_image(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        from rembg import remove

        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)

        # return BGRA image
        output = remove(bgr_np_img, session=self.session)
        return cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)

    def gen_mask(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        from rembg import remove

        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)

        # return BGR image, 255 means foreground, 0 means background
        output = remove(bgr_np_img, session=self.session, only_mask=True)
        return output

    def check_dep(self):
        try:
            import rembg
        except ImportError:
            return (
                "RemoveBG is not installed, please install it first. pip install rembg"
            )
