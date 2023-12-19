import os

import PIL.Image
import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL
from loguru import logger

from lama_cleaner.model.base import DiffusionInpaintModel
from lama_cleaner.schema import Config, ModelType


class SDXL(DiffusionInpaintModel):
    name = "sdxl"
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
                torch_dtype=torch_dtype,
                num_in_channels=num_in_channels,
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype
            )
            self.model = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.model_id_or_path,
                revision="main",
                torch_dtype=torch_dtype,
                use_auth_token=kwargs["hf_access_token"],
                vae=vae,
            )

        # https://huggingface.co/docs/diffusers/v0.7.0/en/api/pipelines/stable_diffusion#diffusers.StableDiffusionInpaintPipeline.enable_attention_slicing
        self.model.enable_attention_slicing()
        # https://huggingface.co/docs/diffusers/v0.7.0/en/optimization/fp16#memory-efficient-attention
        if kwargs.get("enable_xformers", False):
            self.model.enable_xformers_memory_efficient_attention()

        if kwargs.get("cpu_offload", False) and use_gpu:
            logger.info("Enable sequential cpu offload")
            self.model.enable_sequential_cpu_offload(gpu_id=0)
        else:
            self.model = self.model.to(device)
            if kwargs["sd_cpu_textencoder"]:
                logger.warning("Stable Diffusion XL not support run TextEncoder on CPU")

        self.callback = kwargs.pop("callback", None)

    @staticmethod
    def download():
        from diffusers import AutoPipelineForInpainting

        AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        )

    def forward(self, image, mask, config: Config):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """
        self.set_scheduler(config)

        if config.sd_mask_blur != 0:
            k = 2 * config.sd_mask_blur + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)[:, :, np.newaxis]

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
            callback=self.callback,
            height=img_h,
            width=img_w,
            generator=torch.manual_seed(config.sd_seed),
            callback_steps=1,
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    @staticmethod
    def is_downloaded() -> bool:
        # model will be downloaded when app start, and can't switch in frontend settings
        return True
