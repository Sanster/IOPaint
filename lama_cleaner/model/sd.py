import random

import PIL.Image
import cv2
import numpy as np
import torch
from diffusers import PNDMScheduler, DDIMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, \
    EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from loguru import logger

from lama_cleaner.helper import resize_max_size
from lama_cleaner.model.base import InpaintModel
from lama_cleaner.model.utils import torch_gc
from lama_cleaner.schema import Config, SDSampler


class CPUTextEncoderWrapper:
    def __init__(self, text_encoder, torch_dtype):
        self.config = text_encoder.config
        self.text_encoder = text_encoder.to(torch.device('cpu'), non_blocking=True)
        self.text_encoder = self.text_encoder.to(torch.float32, non_blocking=True)
        self.torch_dtype = torch_dtype
        del text_encoder
        torch_gc()

    def __call__(self, x, **kwargs):
        input_device = x.device
        return [self.text_encoder(x.to(self.text_encoder.device), **kwargs)[0].to(input_device).to(self.torch_dtype)]


class SD(InpaintModel):
    pad_mod = 8
    min_size = 512

    def init_model(self, device: torch.device, **kwargs):
        from diffusers.pipelines.stable_diffusion import StableDiffusionInpaintPipeline
        fp16 = not kwargs.get('no_half', False)

        model_kwargs = {"local_files_only": kwargs.get('local_files_only', kwargs['sd_run_local'])}
        if kwargs['disable_nsfw'] or kwargs.get('cpu_offload', False):
            logger.info("Disable Stable Diffusion Model NSFW checker")
            model_kwargs.update(dict(
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False
            ))

        use_gpu = device == torch.device('cuda') and torch.cuda.is_available()
        torch_dtype = torch.float16 if use_gpu and fp16 else torch.float32
        self.model = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id_or_path,
            revision="fp16" if use_gpu and fp16 else "main",
            torch_dtype=torch_dtype,
            use_auth_token=kwargs["hf_access_token"],
            **model_kwargs
        )

        # https://huggingface.co/docs/diffusers/v0.7.0/en/api/pipelines/stable_diffusion#diffusers.StableDiffusionInpaintPipeline.enable_attention_slicing
        self.model.enable_attention_slicing()
        # https://huggingface.co/docs/diffusers/v0.7.0/en/optimization/fp16#memory-efficient-attention
        if kwargs.get('enable_xformers', False):
            self.model.enable_xformers_memory_efficient_attention()

        if kwargs.get('cpu_offload', False) and use_gpu:
            # TODO: gpu_id
            logger.info("Enable sequential cpu offload")
            self.model.enable_sequential_cpu_offload(gpu_id=0)
        else:
            self.model = self.model.to(device)
            if kwargs['sd_cpu_textencoder']:
                logger.info("Run Stable Diffusion TextEncoder on CPU")
                self.model.text_encoder = CPUTextEncoderWrapper(self.model.text_encoder, torch_dtype)

        self.callback = kwargs.pop("callback", None)

    def _scaled_pad_forward(self, image, mask, config: Config):
        longer_side_length = int(config.sd_scale * max(image.shape[:2]))
        origin_size = image.shape[:2]
        downsize_image = resize_max_size(image, size_limit=longer_side_length)
        downsize_mask = resize_max_size(mask, size_limit=longer_side_length)
        logger.info(
            f"Resize image to do sd inpainting: {image.shape} -> {downsize_image.shape}"
        )
        inpaint_result = self._pad_forward(downsize_image, downsize_mask, config)
        # only paste masked area result
        inpaint_result = cv2.resize(
            inpaint_result,
            (origin_size[1], origin_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        original_pixel_indices = mask < 127
        inpaint_result[original_pixel_indices] = image[:, :, ::-1][original_pixel_indices]
        return inpaint_result

    def forward(self, image, mask, config: Config):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """

        scheduler_config = self.model.scheduler.config

        if config.sd_sampler == SDSampler.ddim:
            scheduler = DDIMScheduler.from_config(scheduler_config)
        elif config.sd_sampler == SDSampler.pndm:
            scheduler = PNDMScheduler.from_config(scheduler_config)
        elif config.sd_sampler == SDSampler.k_lms:
            scheduler = LMSDiscreteScheduler.from_config(scheduler_config)
        elif config.sd_sampler == SDSampler.k_euler:
            scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
        elif config.sd_sampler == SDSampler.k_euler_a:
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)
        elif config.sd_sampler == SDSampler.dpm_plus_plus:
            scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
        else:
            raise ValueError(config.sd_sampler)

        self.model.scheduler = scheduler

        seed = config.sd_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    @torch.no_grad()
    def __call__(self, image, mask, config: Config):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        # boxes = boxes_from_mask(mask)
        if config.use_croper:
            crop_img, crop_mask, (l, t, r, b) = self._apply_cropper(image, mask, config)
            crop_image = self._scaled_pad_forward(crop_img, crop_mask, config)
            inpaint_result = image[:, :, ::-1]
            inpaint_result[t:b, l:r, :] = crop_image
        else:
            inpaint_result = self._scaled_pad_forward(image, mask, config)

        return inpaint_result

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
    model_id_or_path = "runwayml/stable-diffusion-inpainting"


class SD2(SD):
    model_id_or_path = "stabilityai/stable-diffusion-2-inpainting"
