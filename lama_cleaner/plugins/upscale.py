import cv2

from lama_cleaner.helper import download_model


class RealESRGANUpscaler:
    name = "RealESRGAN"

    def __init__(self, device):
        super().__init__()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        scale = 4
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        model_md5 = "99ec365d4afad750833258a1a24f44ca"
        model_path = download_model(url, model_md5)

        self.model = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            half=True if "cuda" in str(device) else False,
            tile=640,
            tile_pad=10,
            pre_pad=10,
            device=device,
        )

    def __call__(self, rgb_np_img, files, form):
        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)
        scale = float(form["scale"])
        return self.forward(bgr_np_img, scale)

    def forward(self, bgr_np_img, scale: float):
        # 输出是 BGR
        upsampled = self.model.enhance(bgr_np_img, outscale=scale)[0]
        return upsampled
