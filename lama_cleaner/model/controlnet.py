import PIL.Image
import cv2
import numpy as np
import torch
from diffusers import ControlNetModel
from loguru import logger

from lama_cleaner.const import DIFFUSERS_MODEL_FP16_REVERSION
from lama_cleaner.model.base import DiffusionInpaintModel
from lama_cleaner.model.helper.controlnet_preprocess import (
    make_canny_control_image,
    make_openpose_control_image,
    make_depth_control_image,
    make_inpaint_control_image,
)
from lama_cleaner.model.helper.cpu_text_encoder import CPUTextEncoderWrapper
from lama_cleaner.model.utils import get_scheduler
from lama_cleaner.schema import Config, ModelInfo, ModelType

# 为了兼容性
controlnet_name_map = {
    "control_v11p_sd15_canny": "lllyasviel/control_v11p_sd15_canny",
    "control_v11p_sd15_openpose": "lllyasviel/control_v11p_sd15_openpose",
    "control_v11p_sd15_inpaint": "lllyasviel/control_v11p_sd15_inpaint",
    "control_v11f1p_sd15_depth": "lllyasviel/control_v11f1p_sd15_depth",
}


class ControlNet(DiffusionInpaintModel):
    name = "controlnet"
    pad_mod = 8
    min_size = 512

    def init_model(self, device: torch.device, **kwargs):
        fp16 = not kwargs.get("no_half", False)
        model_info: ModelInfo = kwargs["model_info"]
        sd_controlnet_method = kwargs["sd_controlnet_method"]
        sd_controlnet_method = controlnet_name_map.get(
            sd_controlnet_method, sd_controlnet_method
        )

        self.model_info = model_info
        self.sd_controlnet_method = sd_controlnet_method

        model_kwargs = {}
        if kwargs["disable_nsfw"] or kwargs.get("cpu_offload", False):
            logger.info("Disable Stable Diffusion Model NSFW checker")
            model_kwargs.update(
                dict(
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                )
            )

        use_gpu = device == torch.device("cuda") and torch.cuda.is_available()
        torch_dtype = torch.float16 if use_gpu and fp16 else torch.float32

        if model_info.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SD_INPAINT,
        ]:
            from diffusers import (
                StableDiffusionControlNetInpaintPipeline as PipeClass,
            )
        elif model_info.model_type in [
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ]:
            from diffusers import (
                StableDiffusionXLControlNetInpaintPipeline as PipeClass,
            )

        controlnet = ControlNetModel.from_pretrained(
            sd_controlnet_method, torch_dtype=torch_dtype
        )
        if model_info.is_single_file_diffusers:
            self.model = PipeClass.from_single_file(
                model_info.path, controlnet=controlnet
            ).to(torch_dtype)
        else:
            self.model = PipeClass.from_pretrained(
                model_info.path,
                controlnet=controlnet,
                revision="fp16"
                if (
                    model_info.path in DIFFUSERS_MODEL_FP16_REVERSION
                    and use_gpu
                    and fp16
                )
                else "main",
                torch_dtype=torch_dtype,
                **model_kwargs,
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
                logger.info("Run Stable Diffusion TextEncoder on CPU")
                self.model.text_encoder = CPUTextEncoderWrapper(
                    self.model.text_encoder, torch_dtype
                )

        self.callback = kwargs.pop("callback", None)

    def _get_control_image(self, image, mask):
        if "canny" in self.sd_controlnet_method:
            control_image = make_canny_control_image(image)
        elif "openpose" in self.sd_controlnet_method:
            control_image = make_openpose_control_image(image)
        elif "depth" in self.sd_controlnet_method:
            control_image = make_depth_control_image(image)
        elif "inpaint" in self.sd_controlnet_method:
            control_image = make_inpaint_control_image(image, mask)
        else:
            raise NotImplementedError(f"{self.sd_controlnet_method} not implemented")
        return control_image

    def forward(self, image, mask, config: Config):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """
        scheduler_config = self.model.scheduler.config
        scheduler = get_scheduler(config.sd_sampler, scheduler_config)
        self.model.scheduler = scheduler

        if config.sd_mask_blur != 0:
            k = 2 * config.sd_mask_blur + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)[:, :, np.newaxis]

        img_h, img_w = image.shape[:2]
        control_image = self._get_control_image(image, mask)
        mask_image = PIL.Image.fromarray(mask[:, :, -1], mode="L")
        image = PIL.Image.fromarray(image)

        output = self.model(
            image=image,
            mask_image=mask_image,
            control_image=control_image,
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            num_inference_steps=config.sd_steps,
            guidance_scale=config.sd_guidance_scale,
            output_type="np",
            callback=self.callback,
            height=img_h,
            width=img_w,
            generator=torch.manual_seed(config.sd_seed),
            controlnet_conditioning_scale=config.controlnet_conditioning_scale,
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    @staticmethod
    def is_downloaded() -> bool:
        # model will be downloaded when app start, and can't switch in frontend settings
        return True
