import PIL
import PIL.Image
import cv2
import torch
from diffusers import DiffusionPipeline
from loguru import logger

from lama_cleaner.model.base import DiffusionInpaintModel
from lama_cleaner.model.utils import set_seed
from lama_cleaner.schema import Config


class PaintByExample(DiffusionInpaintModel):
    pad_mod = 8
    min_size = 512

    def init_model(self, device: torch.device, **kwargs):
        fp16 = not kwargs.get('no_half', False)
        use_gpu = device == torch.device('cuda') and torch.cuda.is_available()
        torch_dtype = torch.float16 if use_gpu and fp16 else torch.float32
        model_kwargs = {"local_files_only": kwargs.get('local_files_only', False)}

        if kwargs['disable_nsfw'] or kwargs.get('cpu_offload', False):
            logger.info("Disable Paint By Example Model NSFW checker")
            model_kwargs.update(dict(
                safety_checker=None,
                requires_safety_checker=False
            ))

        self.model = DiffusionPipeline.from_pretrained(
            "Fantasy-Studio/Paint-by-Example",
            torch_dtype=torch_dtype,
            **model_kwargs
        )

        self.model.enable_attention_slicing()
        if kwargs.get('enable_xformers', False):
            self.model.enable_xformers_memory_efficient_attention()

        # TODO: gpu_id
        if kwargs.get('cpu_offload', False) and use_gpu:
            self.model.image_encoder = self.model.image_encoder.to(device)
            self.model.enable_sequential_cpu_offload(gpu_id=0)
        else:
            self.model = self.model.to(device)

    def forward(self, image, mask, config: Config):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """
        set_seed(config.paint_by_example_seed)

        output = self.model(
            image=PIL.Image.fromarray(image),
            mask_image=PIL.Image.fromarray(mask[:, :, -1], mode="L"),
            example_image=config.paint_by_example_example_image,
            num_inference_steps=config.paint_by_example_steps,
            output_type='np.array',
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    def forward_post_process(self, result, image, mask, config):
        if config.paint_by_example_match_histograms:
            result = self._match_histograms(result, image[:, :, ::-1], mask)

        if config.paint_by_example_mask_blur != 0:
            k = 2 * config.paint_by_example_mask_blur + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return result, image, mask

    @staticmethod
    def is_downloaded() -> bool:
        # model will be downloaded when app start, and can't switch in frontend settings
        return True
