import random

import PIL.Image
import cv2
import numpy as np
import torch
from diffusers import PNDMScheduler, DDIMScheduler
from loguru import logger
from transformers import FeatureExtractionMixin, ImageFeatureExtractionMixin

from lama_cleaner.helper import norm_img

from lama_cleaner.model.base import InpaintModel
from lama_cleaner.schema import Config, SDSampler


#
#
# def preprocess_image(image):
#     w, h = image.size
#     w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
#     image = image.resize((w, h), resample=PIL.Image.LANCZOS)
#     image = np.array(image).astype(np.float32) / 255.0
#     image = image[None].transpose(0, 3, 1, 2)
#     image = torch.from_numpy(image)
#     # [-1, 1]
#     return 2.0 * image - 1.0
#
#
# def preprocess_mask(mask):
#     mask = mask.convert("L")
#     w, h = mask.size
#     w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
#     mask = mask.resize((w // 8, h // 8), resample=PIL.Image.NEAREST)
#     mask = np.array(mask).astype(np.float32) / 255.0
#     mask = np.tile(mask, (4, 1, 1))
#     mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
#     mask = 1 - mask  # repaint white, keep black
#     mask = torch.from_numpy(mask)
#     return mask

class DummyFeatureExtractorOutput:
    def __init__(self, pixel_values):
        self.pixel_values = pixel_values

    def to(self, device):
        return self


class DummyFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return DummyFeatureExtractorOutput(torch.empty(0, 3))


class DummySafetyChecker:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, clip_input, images):
        return images, False


class SD(InpaintModel):
    pad_mod = 64  # current diffusers only support 64 https://github.com/huggingface/diffusers/pull/505
    min_size = 512

    def init_model(self, device: torch.device, **kwargs):
        from .sd_pipeline import StableDiffusionInpaintPipeline

        model_kwargs = {"local_files_only": kwargs['sd_run_local']}
        if kwargs['sd_disable_nsfw']:
            logger.info("Disable Stable Diffusion Model NSFW checker")
            model_kwargs.update(dict(
                feature_extractor=DummyFeatureExtractor(),
                safety_checker=DummySafetyChecker(),
            ))

        self.model = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id_or_path,
            revision="fp16" if torch.cuda.is_available() else "main",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_auth_token=kwargs["hf_access_token"],
            **model_kwargs
        )
        # https://huggingface.co/docs/diffusers/v0.3.0/en/api/pipelines/stable_diffusion#diffusers.StableDiffusionInpaintPipeline.enable_attention_slicing
        self.model.enable_attention_slicing()
        self.model = self.model.to(device)

        if kwargs['sd_cpu_textencoder']:
            logger.info("Run Stable Diffusion TextEncoder on CPU")
            self.model.text_encoder = self.model.text_encoder.to(torch.device('cpu'), non_blocking=True)
            self.model.text_encoder = self.model.text_encoder.to(torch.float32, non_blocking=True )

        self.callbacks = kwargs.pop("callbacks", None)

    @torch.cuda.amp.autocast()
    def forward(self, image, mask, config: Config):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """

        # image = norm_img(image)  # [0, 1]
        # image = image * 2 - 1  # [0, 1] -> [-1, 1]

        # resize to latent feature map size
        # h, w = mask.shape[:2]
        # mask = cv2.resize(mask, (h // 8, w // 8), interpolation=cv2.INTER_AREA)
        # mask = norm_img(mask)
        #
        # image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        # mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        if config.sd_sampler == SDSampler.ddim:
            scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
        elif config.sd_sampler == SDSampler.pndm:
            PNDM_kwargs = {
                "tensor_format": "pt",
                "beta_schedule": "scaled_linear",
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "num_train_timesteps": 1000,
                "skip_prk_steps": True,
            }
            scheduler = PNDMScheduler(**PNDM_kwargs)
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

        output = self.model(
            prompt=config.prompt,
            init_image=PIL.Image.fromarray(image),
            mask_image=PIL.Image.fromarray(mask[:, :, -1], mode="L"),
            strength=config.sd_strength,
            num_inference_steps=config.sd_steps,
            guidance_scale=config.sd_guidance_scale,
            output_type="np.array",
            callbacks=self.callbacks,
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
        img_h, img_w = image.shape[:2]

        # boxes = boxes_from_mask(mask)
        if config.use_croper:
            logger.info("use croper")
            l, t, w, h = (
                config.croper_x,
                config.croper_y,
                config.croper_width,
                config.croper_height,
            )
            r = l + w
            b = t + h

            l = max(l, 0)
            r = min(r, img_w)
            t = max(t, 0)
            b = min(b, img_h)

            crop_img = image[t:b, l:r, :]
            crop_mask = mask[t:b, l:r]

            crop_image = self._pad_forward(crop_img, crop_mask, config)

            inpaint_result = image[:, :, ::-1]
            inpaint_result[t:b, l:r, :] = crop_image
        else:
            inpaint_result = self._pad_forward(image, mask, config)

        return inpaint_result

    @staticmethod
    def is_downloaded() -> bool:
        # model will be downloaded when app start, and can't switch in frontend settings
        return True


class SD14(SD):
    model_id_or_path = "CompVis/stable-diffusion-v1-4"


class SD15(SD):
    model_id_or_path = "CompVis/stable-diffusion-v1-5"
