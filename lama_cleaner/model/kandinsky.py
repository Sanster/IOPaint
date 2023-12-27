import PIL.Image
import cv2
import numpy as np
import torch

from lama_cleaner.model.base import DiffusionInpaintModel
from lama_cleaner.model.utils import get_scheduler
from lama_cleaner.schema import Config


class Kandinsky(DiffusionInpaintModel):
    pad_mod = 64
    min_size = 512

    def init_model(self, device: torch.device, **kwargs):
        from diffusers import AutoPipelineForInpainting

        fp16 = not kwargs.get("no_half", False)
        use_gpu = device == torch.device("cuda") and torch.cuda.is_available()
        torch_dtype = torch.float16 if use_gpu and fp16 else torch.float32

        model_kwargs = {
            "torch_dtype": torch_dtype,
        }

        self.model = AutoPipelineForInpainting.from_pretrained(
            self.name, **model_kwargs
        ).to(device)

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
            callback=self.callback,
            generator=generator,
            callback_steps=1,
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output


class Kandinsky22(Kandinsky):
    name = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
