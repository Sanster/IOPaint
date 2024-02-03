import PIL.Image
import cv2
import numpy as np
import torch

from iopaint.const import KANDINSKY22_NAME
from .base import DiffusionInpaintModel
from iopaint.schema import InpaintRequest
from .utils import get_torch_dtype, enable_low_mem, is_local_files_only


class Kandinsky(DiffusionInpaintModel):
    pad_mod = 64
    min_size = 512

    def init_model(self, device: torch.device, **kwargs):
        from diffusers import AutoPipelineForInpainting

        use_gpu, torch_dtype = get_torch_dtype(device, kwargs.get("no_half", False))

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "local_files_only": is_local_files_only(**kwargs),
        }
        self.model = AutoPipelineForInpainting.from_pretrained(
            self.name, **model_kwargs
        ).to(device)
        enable_low_mem(self.model, kwargs.get("low_mem", False))

        self.callback = kwargs.pop("callback", None)

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """
        self.set_scheduler(config)

        generator = torch.manual_seed(config.sd_seed)
        mask = mask.astype(np.float32) / 255
        img_h, img_w = image.shape[:2]

        # kandinsky 没有 strength
        output = self.model(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            image=PIL.Image.fromarray(image),
            mask_image=mask[:, :, 0],
            height=img_h,
            width=img_w,
            num_inference_steps=config.sd_steps,
            guidance_scale=config.sd_guidance_scale,
            output_type="np",
            callback_on_step_end=self.callback,
            generator=generator,
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output


class Kandinsky22(Kandinsky):
    name = KANDINSKY22_NAME
