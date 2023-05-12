import gc

import PIL.Image
import cv2
import numpy as np
import torch
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


def load_from_local_model(local_model_path, torch_dtype, disable_nsfw=True):
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        download_from_original_stable_diffusion_ckpt,
    )
    from diffusers.pipelines.stable_diffusion import StableDiffusionInpaintPipeline

    logger.info(f"Converting {local_model_path} to diffusers pipeline")

    pipe = download_from_original_stable_diffusion_ckpt(
        local_model_path,
        num_in_channels=9,
        from_safetensors=local_model_path.endswith("safetensors"),
        device="cpu",
    )

    inpaint_pipe = StableDiffusionInpaintPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
        safety_checker=None if disable_nsfw else pipe.safety_checker,
        feature_extractor=None if disable_nsfw else pipe.safety_checker,
        requires_safety_checker=not disable_nsfw,
    )

    del pipe
    gc.collect()
    return inpaint_pipe.to(torch_dtype=torch_dtype)


class SD(DiffusionInpaintModel):
    pad_mod = 8
    min_size = 512

    def init_model(self, device: torch.device, **kwargs):
        from diffusers.pipelines.stable_diffusion import StableDiffusionInpaintPipeline

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

        if kwargs.get("sd_local_model_path", None):
            self.model = load_from_local_model(
                kwargs["sd_local_model_path"],
                torch_dtype=torch_dtype,
            )
        else:
            self.model = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id_or_path,
                revision="fp16" if use_gpu and fp16 else "main",
                torch_dtype=torch_dtype,
                use_auth_token=kwargs["hf_access_token"],
                **model_kwargs,
            )

        # https://huggingface.co/docs/diffusers/v0.7.0/en/api/pipelines/stable_diffusion#diffusers.StableDiffusionInpaintPipeline.enable_attention_slicing
        self.model.enable_attention_slicing()
        # https://huggingface.co/docs/diffusers/v0.7.0/en/optimization/fp16#memory-efficient-attention
        if kwargs.get("enable_xformers", False):
            self.model.enable_xformers_memory_efficient_attention()

        if kwargs.get("cpu_offload", False) and use_gpu:
            # TODO: gpu_id
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

        output = self.model(
            image=PIL.Image.fromarray(image),
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            mask_image=PIL.Image.fromarray(mask[:, :, -1], mode="L"),
            num_inference_steps=config.sd_steps,
            guidance_scale=config.sd_guidance_scale,
            output_type="np.array",
            callback=self.callback,
            height=img_h,
            width=img_w,
            generator=torch.manual_seed(config.sd_seed),
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


class SD15(SD):
    name = "sd1.5"
    model_id_or_path = "runwayml/stable-diffusion-inpainting"


class Anything4(SD):
    name = "anything4"
    model_id_or_path = "Sanster/anything-4.0-inpainting"


class RealisticVision14(SD):
    name = "realisticVision1.4"
    model_id_or_path = "Sanster/Realistic_Vision_V1.4-inpainting"


class SD2(SD):
    name = "sd2"
    model_id_or_path = "stabilityai/stable-diffusion-2-inpainting"
