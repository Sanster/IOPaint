import PIL
import PIL.Image
import cv2
import torch
from loguru import logger

from iopaint.helper import decode_base64_to_image
from .base import DiffusionInpaintModel
from iopaint.schema import InpaintRequest


class PaintByExample(DiffusionInpaintModel):
    name = "Fantasy-Studio/Paint-by-Example"
    pad_mod = 8
    min_size = 512

    def init_model(self, device: torch.device, **kwargs):
        from diffusers import DiffusionPipeline

        fp16 = not kwargs.get("no_half", False)
        use_gpu = device == torch.device("cuda") and torch.cuda.is_available()
        torch_dtype = torch.float16 if use_gpu and fp16 else torch.float32
        model_kwargs = {}

        if kwargs["disable_nsfw"] or kwargs.get("cpu_offload", False):
            logger.info("Disable Paint By Example Model NSFW checker")
            model_kwargs.update(
                dict(safety_checker=None, requires_safety_checker=False)
            )

        self.model = DiffusionPipeline.from_pretrained(
            self.name, torch_dtype=torch_dtype, **model_kwargs
        )

        # TODO: gpu_id
        if kwargs.get("cpu_offload", False) and use_gpu:
            self.model.image_encoder = self.model.image_encoder.to(device)
            self.model.enable_sequential_cpu_offload(gpu_id=0)
        else:
            self.model = self.model.to(device)

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """
        if config.paint_by_example_example_image is None:
            raise ValueError("paint_by_example_example_image is required")
        example_image, _, _ = decode_base64_to_image(
            config.paint_by_example_example_image
        )
        output = self.model(
            image=PIL.Image.fromarray(image),
            mask_image=PIL.Image.fromarray(mask[:, :, -1], mode="L"),
            example_image=PIL.Image.fromarray(example_image),
            num_inference_steps=config.sd_steps,
            guidance_scale=config.sd_guidance_scale,
            negative_prompt="out of frame, lowres, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, watermark, signature",
            output_type="np.array",
            generator=torch.manual_seed(config.sd_seed),
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output
