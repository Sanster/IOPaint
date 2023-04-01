import gc

import PIL.Image
import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
)
from loguru import logger

from lama_cleaner.model.base import DiffusionInpaintModel
from lama_cleaner.model.utils import torch_gc, get_scheduler
from lama_cleaner.schema import Config


class CPUTextEncoderWrapper:
    def __init__(self, text_encoder, torch_dtype):
        self.config = text_encoder.config
        self.text_encoder = text_encoder.to(torch.device("cpu"), non_blocking=True)
        self.text_encoder = self.text_encoder.to(torch.float32, non_blocking=True)
        self.torch_dtype = torch_dtype
        del text_encoder
        torch_gc()

    def __call__(self, x, **kwargs):
        input_device = x.device
        return [
            self.text_encoder(x.to(self.text_encoder.device), **kwargs)[0]
            .to(input_device)
            .to(self.torch_dtype)
        ]

    @property
    def dtype(self):
        return self.torch_dtype


NAMES_MAP = {
    "sd1.5": "runwayml/stable-diffusion-inpainting",
    "anything4": "Sanster/anything-4.0-inpainting",
    "realisticVision1.4": "Sanster/Realistic_Vision_V1.4-inpainting",
}


def load_from_local_model(local_model_path, torch_dtype, controlnet):
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        load_pipeline_from_original_stable_diffusion_ckpt,
    )
    from .pipeline import StableDiffusionControlNetInpaintPipeline

    logger.info(f"Converting {local_model_path} to diffusers controlnet pipeline")

    pipe = load_pipeline_from_original_stable_diffusion_ckpt(
        local_model_path,
        num_in_channels=9,
        from_safetensors=local_model_path.endswith("safetensors"),
        device="cpu",
    )

    inpaint_pipe = StableDiffusionControlNetInpaintPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        controlnet=controlnet,
        scheduler=pipe.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    del pipe
    gc.collect()
    return inpaint_pipe.to(torch_dtype)


class ControlNet(DiffusionInpaintModel):
    name = "controlnet"
    pad_mod = 8
    min_size = 512

    def init_model(self, device: torch.device, **kwargs):
        from .pipeline import StableDiffusionControlNetInpaintPipeline

        model_id = NAMES_MAP[kwargs["name"]]
        fp16 = not kwargs.get("no_half", False)

        model_kwargs = {
            "local_files_only": kwargs.get("local_files_only", kwargs["sd_run_local"])
        }
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

        controlnet = ControlNetModel.from_pretrained(
            f"lllyasviel/sd-controlnet-canny", torch_dtype=torch_dtype
        )
        if kwargs.get("sd_local_model_path", None):
            self.model = load_from_local_model(
                kwargs["sd_local_model_path"],
                torch_dtype=torch_dtype,
                controlnet=controlnet,
            )
        else:
            self.model = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                model_id,
                controlnet=controlnet,
                revision="fp16" if use_gpu and fp16 else "main",
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

        canny_image = cv2.Canny(image, 100, 200)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        canny_image = PIL.Image.fromarray(canny_image)
        mask_image = PIL.Image.fromarray(mask[:, :, -1], mode="L")
        image = PIL.Image.fromarray(image)

        output = self.model(
            image=image,
            control_image=canny_image,
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            mask_image=mask_image,
            num_inference_steps=config.sd_steps,
            guidance_scale=config.sd_guidance_scale,
            output_type="np.array",
            callback=self.callback,
            height=img_h,
            width=img_w,
            generator=torch.manual_seed(config.sd_seed),
            controlnet_conditioning_scale=config.controlnet_conditioning_scale,
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    def forward_post_process(self, result, image, mask, config):
        if config.sd_match_histograms:
            result = self._match_histograms(result, image[:, :, ::-1], mask)

        if config.sd_mask_blur != 0:
            k = 2 * config.sd_mask_blur + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return result, image, mask

    @staticmethod
    def is_downloaded() -> bool:
        # model will be downloaded when app start, and can't switch in frontend settings
        return True
