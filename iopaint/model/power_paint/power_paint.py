from PIL import Image
import PIL.Image
import cv2
import torch
from loguru import logger

from ..base import DiffusionInpaintModel
from ..helper.cpu_text_encoder import CPUTextEncoderWrapper
from ..utils import (
    handle_from_pretrained_exceptions,
    get_torch_dtype,
    enable_low_mem,
    is_local_files_only,
)
from iopaint.schema import InpaintRequest
from .powerpaint_tokenizer import add_task_to_prompt
from ...const import POWERPAINT_NAME


class PowerPaint(DiffusionInpaintModel):
    name = POWERPAINT_NAME
    pad_mod = 8
    min_size = 512
    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

    def init_model(self, device: torch.device, **kwargs):
        from .pipeline_powerpaint import StableDiffusionInpaintPipeline
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

        self.model = handle_from_pretrained_exceptions(
            StableDiffusionInpaintPipeline.from_pretrained,
            pretrained_model_name_or_path=self.name,
            variant="fp16",
            torch_dtype=torch_dtype,
            **model_kwargs,
        )
        self.model.tokenizer = PowerPaintTokenizer(self.model.tokenizer)

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

        img_h, img_w = image.shape[:2]
        promptA, promptB, negative_promptA, negative_promptB = add_task_to_prompt(
            config.prompt, config.negative_prompt, config.powerpaint_task
        )

        output = self.model(
            image=PIL.Image.fromarray(image),
            promptA=promptA,
            promptB=promptB,
            tradoff=config.fitting_degree,
            tradoff_nag=config.fitting_degree,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            mask_image=PIL.Image.fromarray(mask[:, :, -1], mode="L"),
            num_inference_steps=config.sd_steps,
            strength=config.sd_strength,
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
