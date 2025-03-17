import PIL.Image
import cv2
import torch
from loguru import logger
import numpy as np

from ..base import DiffusionInpaintModel
from ..helper.cpu_text_encoder import CPUTextEncoderWrapper
from ..original_sd_configs import get_config_files
from ..utils import (
    handle_from_pretrained_exceptions,
    get_torch_dtype,
    enable_low_mem,
    is_local_files_only,
)
from .brushnet import BrushNetModel
from .brushnet_unet_forward import brushnet_unet_forward
from .unet_2d_blocks import (
    CrossAttnDownBlock2D_forward,
    DownBlock2D_forward,
    CrossAttnUpBlock2D_forward,
    UpBlock2D_forward,
)
from ...schema import InpaintRequest, ModelType
from ...const import SDXL_BRUSHNET_CHOICES


class BrushNetXLWrapper(DiffusionInpaintModel):
    pad_mod = 8
    min_size = 1024
    support_brushnet = True
    support_lcm_lora = False

    def init_model(self, device: torch.device, **kwargs):
        from .pipeline_brushnet_sd_xl import StableDiffusionXLBrushNetPipeline

        self.model_info = kwargs["model_info"]
        self.brushnet_xl_method = SDXL_BRUSHNET_CHOICES[0]
        # self.brushnet_xl_method = kwargs["brushnet_xl_method"]

        use_gpu, torch_dtype = get_torch_dtype(device, kwargs.get("no_half", False))
        self.torch_dtype = torch_dtype

        model_kwargs = {
            **kwargs.get("pipe_components", {}),
            "local_files_only": is_local_files_only(**kwargs),
        }
        self.local_files_only = model_kwargs["local_files_only"]

        disable_nsfw_checker = kwargs["disable_nsfw"] or kwargs.get(
            "cpu_offload", False
        )
        if disable_nsfw_checker:
            logger.info("Disable Stable Diffusion Model NSFW checker")
            model_kwargs.update(
                dict(
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                )
            )

        logger.info(f"Loading BrushNet model from {self.brushnet_xl_method}")
        brushnet = BrushNetModel.from_pretrained(
            self.brushnet_xl_method, torch_dtype=torch_dtype
        )

        if self.model_info.is_single_file_diffusers:
            if self.model_info.model_type == ModelType.DIFFUSERS_SD:
                model_kwargs["num_in_channels"] = 4
            else:
                model_kwargs["num_in_channels"] = 9

            self.model = StableDiffusionXLBrushNetPipeline.from_single_file(
                self.model_id_or_path,
                torch_dtype=torch_dtype,
                load_safety_checker=not disable_nsfw_checker,
                original_config_file=get_config_files()["v1"],
                brushnet=brushnet,
                **model_kwargs,
            )
        else:
            self.model = handle_from_pretrained_exceptions(
                StableDiffusionXLBrushNetPipeline.from_pretrained,
                pretrained_model_name_or_path=self.model_id_or_path,
                variant="fp16",
                torch_dtype=torch_dtype,
                brushnet=brushnet,
                **model_kwargs,
            )

        enable_low_mem(self.model, kwargs.get("low_mem", False))

        if kwargs.get("cpu_offload", False) and use_gpu:
            logger.info("Enable sequential cpu offload")
            self.model.enable_sequential_cpu_offload(gpu_id=0)
        else:
            self.model = self.model.to(device)
            if kwargs["sd_cpu_textencoder"]:
                logger.info("Run Stable Diffusion TextEncoder on CPU")
                self.model.text_encoder = CPUTextEncoderWrapper(
                    self.model.text_encoder, torch_dtype
                )

        self.callback = kwargs.pop("callback", None)

        # Monkey patch the forward method of the UNet to use the brushnet_unet_forward method
        self.model.unet.forward = brushnet_unet_forward.__get__(
            self.model.unet, self.model.unet.__class__
        )

        for down_block in self.model.brushnet.down_blocks:
            down_block.forward = DownBlock2D_forward.__get__(
                down_block, down_block.__class__
            )
        for up_block in self.model.brushnet.up_blocks:
            up_block.forward = UpBlock2D_forward.__get__(up_block, up_block.__class__)

        # Monkey patch unet down_blocks to use CrossAttnDownBlock2D_forward
        for down_block in self.model.unet.down_blocks:
            if down_block.__class__.__name__ == "CrossAttnDownBlock2D":
                down_block.forward = CrossAttnDownBlock2D_forward.__get__(
                    down_block, down_block.__class__
                )
            else:
                down_block.forward = DownBlock2D_forward.__get__(
                    down_block, down_block.__class__
                )

        for up_block in self.model.unet.up_blocks:
            if up_block.__class__.__name__ == "CrossAttnUpBlock2D":
                up_block.forward = CrossAttnUpBlock2D_forward.__get__(
                    up_block, up_block.__class__
                )
            else:
                up_block.forward = UpBlock2D_forward.__get__(
                    up_block, up_block.__class__
                )

    def switch_brushnet_method(self, new_method: str):
        self.brushnet_method = new_method
        brushnet_xl = BrushNetModel.from_pretrained(
            new_method,
            local_files_only=self.local_files_only,
            torch_dtype=self.torch_dtype,
        ).to(self.model.device)
        self.model.brushnet = brushnet_xl

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """
        self.set_scheduler(config)

        img_h, img_w = image.shape[:2]
        normalized_mask = mask[:, :].astype("float32") / 255.0
        image = image * (1 - normalized_mask)
        image = image.astype(np.uint8)
        output = self.model(
            image=PIL.Image.fromarray(image),
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            mask=PIL.Image.fromarray(mask[:, :, -1], mode="L").convert("RGB"),
            num_inference_steps=config.sd_steps,
            # strength=config.sd_strength,
            guidance_scale=config.sd_guidance_scale,
            output_type="np",
            callback_on_step_end=self.callback,
            height=img_h,
            width=img_w,
            generator=torch.manual_seed(config.sd_seed),
            brushnet_conditioning_scale=config.brushnet_conditioning_scale,
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output
