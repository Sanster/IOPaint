import cv2
from loguru import logger

from lama_cleaner.helper import download_model
from lama_cleaner.plugins.base_plugin import BasePlugin


class GFPGANPlugin(BasePlugin):
    name = "GFPGAN"

    def __init__(self, device, upscaler=None):
        super().__init__()
        from .gfpganer import MyGFPGANer

        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        model_md5 = "94d735072630ab734561130a47bc44f8"
        model_path = download_model(url, model_md5)
        logger.info(f"GFPGAN model path: {model_path}")

        import facexlib

        if hasattr(facexlib.detection.retinaface, "device"):
            facexlib.detection.retinaface.device = device

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

    def __call__(self, rgb_np_img, files, form):
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

    def check_dep(self):
        try:
            import gfpgan
        except ImportError:
            return (
                "gfpgan is not installed, please install it first. pip install gfpgan"
            )
