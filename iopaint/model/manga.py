import os
import random

import cv2
import numpy as np
import torch
import time
from loguru import logger

from iopaint.helper import get_cache_path_by_url, load_jit_model, download_model
from .base import InpaintModel
from iopaint.schema import InpaintRequest


MANGA_INPAINTOR_MODEL_URL = os.environ.get(
    "MANGA_INPAINTOR_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/manga/manga_inpaintor.jit",
)
MANGA_INPAINTOR_MODEL_MD5 = os.environ.get(
    "MANGA_INPAINTOR_MODEL_MD5", "7d8b269c4613b6b3768af714610da86c"
)

MANGA_LINE_MODEL_URL = os.environ.get(
    "MANGA_LINE_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/manga/erika.jit",
)
MANGA_LINE_MODEL_MD5 = os.environ.get(
    "MANGA_LINE_MODEL_MD5", "0c926d5a4af8450b0d00bc5b9a095644"
)


class Manga(InpaintModel):
    name = "manga"
    pad_mod = 16
    is_erase_model = True

    def init_model(self, device, **kwargs):
        self.inpaintor_model = load_jit_model(
            MANGA_INPAINTOR_MODEL_URL, device, MANGA_INPAINTOR_MODEL_MD5
        )
        self.line_model = load_jit_model(
            MANGA_LINE_MODEL_URL, device, MANGA_LINE_MODEL_MD5
        )
        self.seed = 42

    @staticmethod
    def download():
        download_model(MANGA_INPAINTOR_MODEL_URL, MANGA_INPAINTOR_MODEL_MD5)
        download_model(MANGA_LINE_MODEL_URL, MANGA_LINE_MODEL_MD5)

    @staticmethod
    def is_downloaded() -> bool:
        model_paths = [
            get_cache_path_by_url(MANGA_INPAINTOR_MODEL_URL),
            get_cache_path_by_url(MANGA_LINE_MODEL_URL),
        ]
        return all([os.path.exists(it) for it in model_paths])

    def forward(self, image, mask, config: InpaintRequest):
        """
        image: [H, W, C] RGB
        mask: [H, W, 1]
        return: BGR IMAGE
        """
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_img = torch.from_numpy(
            gray_img[np.newaxis, np.newaxis, :, :].astype(np.float32)
        ).to(self.device)
        start = time.time()
        lines = self.line_model(gray_img)
        torch.cuda.empty_cache()
        lines = torch.clamp(lines, 0, 255)
        logger.info(f"erika_model time: {time.time() - start}")

        mask = torch.from_numpy(mask[np.newaxis, :, :, :]).to(self.device)
        mask = mask.permute(0, 3, 1, 2)
        mask = torch.where(mask > 0.5, 1.0, 0.0)
        noise = torch.randn_like(mask)
        ones = torch.ones_like(mask)

        gray_img = gray_img / 255 * 2 - 1.0
        lines = lines / 255 * 2 - 1.0

        start = time.time()
        inpainted_image = self.inpaintor_model(gray_img, lines, mask, noise, ones)
        logger.info(f"image_inpaintor_model time: {time.time() - start}")

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = (cur_res * 127.5 + 127.5).astype(np.uint8)
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_GRAY2BGR)
        return cur_res
