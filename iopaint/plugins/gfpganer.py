import os

import torch
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from gfpgan import GFPGANv1Clean, GFPGANer
from torch.hub import get_dir


class MyGFPGANer(GFPGANer):
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(
        self,
        model_path,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=None,
    ):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        # initialize the GFP-GAN
        if arch == "clean":
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "RestoreFormer":
            from gfpgan.archs.restoreformer_arch import RestoreFormer

            self.gfpgan = RestoreFormer()

        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
            model_rootpath=model_dir,
        )

        loadnet = torch.load(model_path)
        if "params_ema" in loadnet:
            keyname = "params_ema"
        else:
            keyname = "params"
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)
