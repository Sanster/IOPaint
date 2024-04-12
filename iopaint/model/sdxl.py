import os

import PIL.Image
import cv2
import torch
from diffusers import AutoencoderKL
from loguru import logger

from iopaint.schema import InpaintRequest, ModelType

from .base import DiffusionInpaintModel
from .helper.cpu_text_encoder import CPUTextEncoderWrapper
from .original_sd_configs import get_config_files
from .utils import (
    handle_from_pretrained_exceptions,
    get_torch_dtype,
    enable_low_mem,
    is_local_files_only,
)


class SDXL(DiffusionInpaintModel):
    name = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    pad_mod = 8
    min_size = 512
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
    model_id_or_path = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

    def init_model(self, device: torch.device, **kwargs):
        from diffusers.pipelines import StableDiffusionXLInpaintPipeline

        use_gpu, torch_dtype = get_torch_dtype(device, kwargs.get("no_half", False))

        if self.model_info.model_type == ModelType.DIFFUSERS_SDXL:
            num_in_channels = 4
        else:
            num_in_channels = 9

        if os.path.isfile(self.model_id_or_path):
            self.model = StableDiffusionXLInpaintPipeline.from_single_file(
                self.model_id_or_path,
                torch_dtype=torch_dtype,
                num_in_channels=num_in_channels,
                load_safety_checker=False,
                original_config_file=get_config_files()['xl'],
            )
        else:
            model_kwargs = {
                **kwargs.get("pipe_components", {}),
                "local_files_only": is_local_files_only(**kwargs),
            }
            if "vae" not in model_kwargs:
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype
                )
                model_kwargs["vae"] = vae
            self.model = handle_from_pretrained_exceptions(
                StableDiffusionXLInpaintPipeline.from_pretrained,
                pretrained_model_name_or_path=self.model_id_or_path,
                torch_dtype=torch_dtype,
                variant="fp16",
                **model_kwargs
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
                self.model.text_encoder_2 = CPUTextEncoderWrapper(
                    self.model.text_encoder_2, torch_dtype
                )

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
