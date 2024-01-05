import os

import PIL.Image
import cv2
import torch
from diffusers import AutoencoderKL
from loguru import logger

from iopaint.schema import InpaintRequest, ModelType

from .base import DiffusionInpaintModel
from .utils import handle_from_pretrained_exceptions


class SDXL(DiffusionInpaintModel):
    name = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    pad_mod = 8
    min_size = 512
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
    model_id_or_path = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

    def init_model(self, device: torch.device, **kwargs):
        from diffusers.pipelines import StableDiffusionXLInpaintPipeline

        fp16 = not kwargs.get("no_half", False)

        use_gpu = device == torch.device("cuda") and torch.cuda.is_available()
        torch_dtype = torch.float16 if use_gpu and fp16 else torch.float32

        if self.model_info.model_type == ModelType.DIFFUSERS_SDXL:
            num_in_channels = 4
        else:
            num_in_channels = 9

        if os.path.isfile(self.model_id_or_path):
            self.model = StableDiffusionXLInpaintPipeline.from_single_file(
                self.model_id_or_path,
                dtype=torch_dtype,
                num_in_channels=num_in_channels,
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype
            )
            self.model = handle_from_pretrained_exceptions(
                StableDiffusionXLInpaintPipeline.from_pretrained,
                pretrained_model_name_or_path=self.model_id_or_path,
                torch_dtype=torch_dtype,
                vae=vae,
                variant="fp16",
            )

        if kwargs.get("cpu_offload", False) and use_gpu:
            logger.info("Enable sequential cpu offload")
            self.model.enable_sequential_cpu_offload(gpu_id=0)
        else:
            self.model = self.model.to(device)
            if kwargs["sd_cpu_textencoder"]:
                logger.warning("Stable Diffusion XL not support run TextEncoder on CPU")

        self.callback = kwargs.pop("callback", None)

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """
        self.set_scheduler(config)

        img_h, img_w = image.shape[:2]

        output = self.model(
            image=PIL.Image.fromarray(image),
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            mask_image=PIL.Image.fromarray(mask[:, :, -1], mode="L"),
            num_inference_steps=config.sd_steps,
            strength=0.999 if config.sd_strength == 1.0 else config.sd_strength,
            guidance_scale=config.sd_guidance_scale,
            output_type="np",
            callback_on_step_end=self.callback,
            height=img_h,
            width=img_w,
            generator=torch.manual_seed(config.sd_seed),
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output
