import cv2
import numpy as np
from loguru import logger

from iopaint.helper import download_model
from iopaint.plugins.base_plugin import BasePlugin
from iopaint.schema import RunPluginRequest


class GFPGANPlugin(BasePlugin):
    name = "GFPGAN"
    support_gen_image = True

    def __init__(self, device, upscaler=None):
        super().__init__()
        from .gfpganer import MyGFPGANer

        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        model_md5 = "94d735072630ab734561130a47bc44f8"
        model_path = download_model(url, model_md5)
        logger.info(f"GFPGAN model path: {model_path}")

        # Use GFPGAN for face enhancement
        self.face_enhancer = MyGFPGANer(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            device=device,
            bg_upsampler=upscaler.model if upscaler is not None else None,
        )
        self.face_enhancer.face_helper.face_det.mean_tensor.to(device)
        self.face_enhancer.face_helper.face_det = (
            self.face_enhancer.face_helper.face_det.to(device)
        )

    def gen_image(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        weight = 0.5
        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)
        logger.info(f"GFPGAN input shape: {bgr_np_img.shape}")
        _, _, bgr_output = self.face_enhancer.enhance(
            bgr_np_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=weight,
        )
        logger.info(f"GFPGAN output shape: {bgr_output.shape}")

        # try:
        #     if scale != 2:
        #         interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
        #         h, w = img.shape[0:2]
        #         output = cv2.resize(
        #             output,
        #             (int(w * scale / 2), int(h * scale / 2)),
        #             interpolation=interpolation,
        #         )
        # except Exception as error:
        #     print("wrong scale input.", error)
        return bgr_output
