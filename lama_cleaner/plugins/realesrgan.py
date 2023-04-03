from enum import Enum

import cv2
from loguru import logger

from lama_cleaner.const import RealESRGANModelName
from lama_cleaner.helper import download_model
from lama_cleaner.plugins.base_plugin import BasePlugin


class RealESRGANUpscaler(BasePlugin):
    name = "RealESRGAN"

    def __init__(self, name, device, no_half=False):
        super().__init__()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact

        REAL_ESRGAN_MODELS = {
            RealESRGANModelName.realesr_general_x4v3: {
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                "scale": 4,
                "model": lambda: SRVGGNetCompact(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_conv=32,
                    upscale=4,
                    act_type="prelu",
                ),
                "model_md5": "91a7644643c884ee00737db24e478156",
            },
            RealESRGANModelName.RealESRGAN_x4plus: {
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "scale": 4,
                "model": lambda: RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                ),
                "model_md5": "99ec365d4afad750833258a1a24f44ca",
            },
            RealESRGANModelName.RealESRGAN_x4plus_anime_6B: {
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                "scale": 4,
                "model": lambda: RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=6,
                    num_grow_ch=32,
                    scale=4,
                ),
                "model_md5": "d58ce384064ec1591c2ea7b79dbf47ba",
            },
        }
        if name not in REAL_ESRGAN_MODELS:
            raise ValueError(f"Unknown RealESRGAN model name: {name}")
        model_info = REAL_ESRGAN_MODELS[name]

        model_path = download_model(model_info["url"], model_info["model_md5"])
        logger.info(f"RealESRGAN model path: {model_path}")

        self.model = RealESRGANer(
            scale=model_info["scale"],
            model_path=model_path,
            model=model_info["model"](),
            half=True if "cuda" in str(device) and not no_half else False,
            tile=512,
            tile_pad=10,
            pre_pad=10,
            device=device,
        )

    def __call__(self, rgb_np_img, files, form):
        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)
        scale = float(form["upscale"])
        logger.info(f"RealESRGAN input shape: {bgr_np_img.shape}, scale: {scale}")
        result = self.forward(bgr_np_img, scale)
        logger.info(f"RealESRGAN output shape: {result.shape}")
        return result

    def forward(self, bgr_np_img, scale: float):
        # 输出是 BGR
        upsampled = self.model.enhance(bgr_np_img, outscale=scale)[0]
        return upsampled

    def check_dep(self):
        try:
            import realesrgan
        except ImportError:
            return "RealESRGAN is not installed, please install it first. pip install realesrgan"
