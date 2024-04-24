import PIL.Image
import cv2
import torch
from loguru import logger
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np

from ..base import DiffusionInpaintModel
from ..helper.cpu_text_encoder import CPUTextEncoderWrapper
from ..utils import (
    get_torch_dtype,
    enable_low_mem,
    is_local_files_only,
    handle_from_pretrained_exceptions,
)
from .powerpaint_tokenizer import task_to_prompt
from iopaint.schema import InpaintRequest
from .v2.BrushNet_CA import BrushNetModel
from .v2.unet_2d_condition import UNet2DConditionModel


class PowerPaintV2(DiffusionInpaintModel):
    pad_mod = 8
    min_size = 512
    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
    hf_model_id = "Sanster/PowerPaint_v2"

    def init_model(self, device: torch.device, **kwargs):
        from .v2.pipeline_PowerPaint_Brushnet_CA import (
            StableDiffusionPowerPaintBrushNetPipeline,
        )
        from .powerpaint_tokenizer import PowerPaintTokenizer

        use_gpu, torch_dtype = get_torch_dtype(device, kwargs.get("no_half", False))
        model_kwargs = {"local_files_only": is_local_files_only(**kwargs)}
        if kwargs["disable_nsfw"] or kwargs.get("cpu_offload", False):
            logger.info("Disable Stable Diffusion Model NSFW checker")
            model_kwargs.update(
                dict(
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                )
            )

        text_encoder_brushnet = CLIPTextModel.from_pretrained(
            self.hf_model_id,
            subfolder="text_encoder_brushnet",
            variant="fp16",
            torch_dtype=torch_dtype,
            local_files_only=model_kwargs["local_files_only"],
        )
        unet = handle_from_pretrained_exceptions(
            UNet2DConditionModel.from_pretrained,
            pretrained_model_name_or_path=self.model_id_or_path,
            subfolder="unet",
            variant="fp16",
            torch_dtype=torch_dtype,
            local_files_only=model_kwargs["local_files_only"],
        )
        brushnet = BrushNetModel.from_pretrained(
            self.hf_model_id,
            subfolder="PowerPaint_Brushnet",
            variant="fp16",
            torch_dtype=torch_dtype,
            local_files_only=model_kwargs["local_files_only"],
        )
        pipe = handle_from_pretrained_exceptions(
            StableDiffusionPowerPaintBrushNetPipeline.from_pretrained,
            pretrained_model_name_or_path=self.model_id_or_path,
            torch_dtype=torch_dtype,
            unet=unet,
            brushnet=brushnet,
            text_encoder_brushnet=text_encoder_brushnet,
            variant="fp16",
            **model_kwargs,
        )
        pipe.tokenizer = PowerPaintTokenizer(
            CLIPTokenizer.from_pretrained(self.hf_model_id, subfolder="tokenizer")
        )
        self.model = pipe

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

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """
        self.set_scheduler(config)

        image = image * (1 - mask / 255.0)
        img_h, img_w = image.shape[:2]

        image = PIL.Image.fromarray(image.astype(np.uint8))
        mask = PIL.Image.fromarray(mask[:, :, -1], mode="L").convert("RGB")

        promptA, promptB, negative_promptA, negative_promptB = task_to_prompt(
            config.powerpaint_task
        )

        output = self.model(
            image=image,
            mask=mask,
            promptA=promptA,
            promptB=promptB,
            promptU=config.prompt,
            tradoff=config.fitting_degree,
            tradoff_nag=config.fitting_degree,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            negative_promptU=config.negative_prompt,
            num_inference_steps=config.sd_steps,
            # strength=config.sd_strength,
            brushnet_conditioning_scale=1.0,
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
