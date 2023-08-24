import gc

import PIL.Image
import cv2
import numpy as np
import torch
from diffusers import ControlNetModel
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

NATIVE_NAMES_MAP = {
    "sd1.5": "runwayml/stable-diffusion-v1-5",
    "anything4": "andite/anything-v4.0",
    "realisticVision1.4": "SG161222/Realistic_Vision_V1.4",
}


def make_inpaint_condition(image, image_mask):
    """
    image: [H, W, C] RGB
    mask: [H, W, 1] 255 means area to repaint
    """
    image = image.astype(np.float32) / 255.0
    image[image_mask[:, :, -1] > 128] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def load_from_local_model(
    local_model_path, torch_dtype, controlnet, pipe_class, is_native_control_inpaint
):
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        download_from_original_stable_diffusion_ckpt,
    )

    logger.info(f"Converting {local_model_path} to diffusers controlnet pipeline")

    try:
        pipe = download_from_original_stable_diffusion_ckpt(
            local_model_path,
            num_in_channels=4 if is_native_control_inpaint else 9,
            from_safetensors=local_model_path.endswith("safetensors"),
            device="cpu",
            load_safety_checker=False,
        )
    except Exception as e:
        err_msg = str(e)
        logger.exception(e)
        if is_native_control_inpaint and "[320, 9, 3, 3]" in err_msg:
            logger.error(
                "control_v11p_sd15_inpaint method requires normal SD model, not inpainting SD model"
            )
        if not is_native_control_inpaint and "[320, 4, 3, 3]" in err_msg:
            logger.error(
                f"{controlnet.config['_name_or_path']} method requires inpainting SD model, "
                f"you can convert any SD model to inpainting model in AUTO1111: \n"
                f"https://www.reddit.com/r/StableDiffusion/comments/zyi24j/how_to_turn_any_model_into_an_inpainting_model/"
            )
        exit(-1)

    inpaint_pipe = pipe_class(
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
    return inpaint_pipe.to(torch_dtype=torch_dtype)


class ControlNet(DiffusionInpaintModel):
    name = "controlnet"
    pad_mod = 8
    min_size = 512

    def init_model(self, device: torch.device, **kwargs):
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

        sd_controlnet_method = kwargs["sd_controlnet_method"]
        self.sd_controlnet_method = sd_controlnet_method

        if sd_controlnet_method == "control_v11p_sd15_inpaint":
            from diffusers import StableDiffusionControlNetPipeline as PipeClass

            self.is_native_control_inpaint = True
        else:
            from .pipeline import StableDiffusionControlNetInpaintPipeline as PipeClass

            self.is_native_control_inpaint = False

        if self.is_native_control_inpaint:
            model_id = NATIVE_NAMES_MAP[kwargs["name"]]
        else:
            model_id = NAMES_MAP[kwargs["name"]]

        controlnet = ControlNetModel.from_pretrained(
            f"lllyasviel/{sd_controlnet_method}", torch_dtype=torch_dtype
        )
        self.is_local_sd_model = False
        if kwargs.get("sd_local_model_path", None):
            self.is_local_sd_model = True
            self.model = load_from_local_model(
                kwargs["sd_local_model_path"],
                torch_dtype=torch_dtype,
                controlnet=controlnet,
                pipe_class=PipeClass,
                is_native_control_inpaint=self.is_native_control_inpaint,
            )
        else:
            self.model = PipeClass.from_pretrained(
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

        if self.is_native_control_inpaint:
            control_image = make_inpaint_condition(image, mask)
            output = self.model(
                prompt=config.prompt,
                image=control_image,
                height=img_h,
                width=img_w,
                num_inference_steps=config.sd_steps,
                guidance_scale=config.sd_guidance_scale,
                controlnet_conditioning_scale=config.controlnet_conditioning_scale,
                negative_prompt=config.negative_prompt,
                generator=torch.manual_seed(config.sd_seed),
                output_type="np.array",
                callback=self.callback,
            ).images[0]
        else:
            if "canny" in self.sd_controlnet_method:
                canny_image = cv2.Canny(image, 100, 200)
                canny_image = canny_image[:, :, None]
                canny_image = np.concatenate(
                    [canny_image, canny_image, canny_image], axis=2
                )
                canny_image = PIL.Image.fromarray(canny_image)
                control_image = canny_image
            elif "openpose" in self.sd_controlnet_method:
                from controlnet_aux import OpenposeDetector

                processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                control_image = processor(image, hand_and_face=True)
            elif "depth" in self.sd_controlnet_method:
                from transformers import pipeline

                depth_estimator = pipeline("depth-estimation")
                depth_image = depth_estimator(PIL.Image.fromarray(image))["depth"]
                depth_image = np.array(depth_image)
                depth_image = depth_image[:, :, None]
                depth_image = np.concatenate(
                    [depth_image, depth_image, depth_image], axis=2
                )
                control_image = PIL.Image.fromarray(depth_image)
            else:
                raise NotImplementedError(
                    f"{self.sd_controlnet_method} not implemented"
                )

            mask_image = PIL.Image.fromarray(mask[:, :, -1], mode="L")
            image = PIL.Image.fromarray(image)

            output = self.model(
                image=image,
                control_image=control_image,
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
