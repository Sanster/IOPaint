import torch
from huggingface_hub import hf_hub_download

from iopaint.const import ANYTEXT_NAME
from iopaint.model.anytext.anytext_pipeline import AnyTextPipeline
from iopaint.model.base import DiffusionInpaintModel
from iopaint.model.utils import get_torch_dtype, is_local_files_only
from iopaint.schema import InpaintRequest


class AnyText(DiffusionInpaintModel):
    name = ANYTEXT_NAME
    pad_mod = 64
    is_erase_model = False

    @staticmethod
    def download(local_files_only=False):
        hf_hub_download(
            repo_id=ANYTEXT_NAME,
            filename="model_index.json",
            local_files_only=local_files_only,
        )
        ckpt_path = hf_hub_download(
            repo_id=ANYTEXT_NAME,
            filename="pytorch_model.fp16.safetensors",
            local_files_only=local_files_only,
        )
        font_path = hf_hub_download(
            repo_id=ANYTEXT_NAME,
            filename="SourceHanSansSC-Medium.otf",
            local_files_only=local_files_only,
        )
        return ckpt_path, font_path

    def init_model(self, device, **kwargs):
        local_files_only = is_local_files_only(**kwargs)
        ckpt_path, font_path = self.download(local_files_only)
        use_gpu, torch_dtype = get_torch_dtype(device, kwargs.get("no_half", False))
        self.model = AnyTextPipeline(
            ckpt_path=ckpt_path,
            font_path=font_path,
            device=device,
            use_fp16=torch_dtype == torch.float16,
        )
        self.callback = kwargs.pop("callback", None)

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to inpainting
        return: BGR IMAGE
        """
        height, width = image.shape[:2]
        mask = mask.astype("float32") / 255.0
        masked_image = image * (1 - mask)

        # list of rgb ndarray
        results, rtn_code, rtn_warning = self.model(
            image=image,
            masked_image=masked_image,
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            num_inference_steps=config.sd_steps,
            strength=config.sd_strength,
            guidance_scale=config.sd_guidance_scale,
            height=height,
            width=width,
            seed=config.sd_seed,
            sort_priority="y",
            callback=self.callback
        )
        inpainted_rgb_image = results[0][..., ::-1]
        return inpainted_rgb_image
