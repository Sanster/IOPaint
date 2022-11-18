import os
import random

import cv2
import numpy as np
import torch
import time
from loguru import logger

from lama_cleaner.helper import get_cache_path_by_url, load_jit_model
from lama_cleaner.model.base import InpaintModel
from lama_cleaner.schema import Config

# def norm(np_img):
#     return np_img / 255 * 2 - 1.0
#
#
# @torch.no_grad()
# def run():
#     name = 'manga_1080x740.jpg'
#     img_p = f'/Users/qing/code/github/MangaInpainting/examples/test/imgs/{name}'
#     mask_p = f'/Users/qing/code/github/MangaInpainting/examples/test/masks/mask_{name}'
#     erika_model = torch.jit.load('erika.jit')
#     manga_inpaintor_model = torch.jit.load('manga_inpaintor.jit')
#
#     img = cv2.imread(img_p)
#     gray_img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
#     mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
#
#     kernel = np.ones((9, 9), dtype=np.uint8)
#     mask = cv2.dilate(mask, kernel, 2)
#     # cv2.imwrite("mask.jpg", mask)
#     # cv2.imshow('dilated_mask', cv2.hconcat([mask, dilated_mask]))
#     # cv2.waitKey(0)
#     # exit()
#
#     # img = pad(img)
#     gray_img = pad(gray_img).astype(np.float32)
#     mask = pad(mask)
#
#     # pad_mod = 16
#     import time
#     start = time.time()
#     y = erika_model(torch.from_numpy(gray_img[np.newaxis, np.newaxis, :, :]))
#     y = torch.clamp(y, 0, 255)
#     lines = y.cpu().numpy()
#     print(f"erika_model time: {time.time() - start}")
#
#     cv2.imwrite('lines.png', lines[0][0])
#
#     start = time.time()
#     masks = torch.from_numpy(mask[np.newaxis, np.newaxis, :, :])
#     masks = torch.where(masks > 0.5, torch.tensor(1.0), torch.tensor(0.0))
#     noise = torch.randn_like(masks)
#
#     images = torch.from_numpy(norm(gray_img)[np.newaxis, np.newaxis, :, :])
#     lines = torch.from_numpy(norm(lines))
#
#     outputs = manga_inpaintor_model(images, lines, masks, noise)
#     print(f"manga_inpaintor_model time: {time.time() - start}")
#
#     outputs_merged = (outputs * masks) + (images * (1 - masks))
#     outputs_merged = outputs_merged * 127.5 + 127.5
#     outputs_merged = outputs_merged.permute(0, 2, 3, 1)[0].detach().cpu().numpy().astype(np.uint8)
#     cv2.imwrite(f'output_{name}', outputs_merged)


MANGA_INPAINTOR_MODEL_URL = os.environ.get(
    "MANGA_INPAINTOR_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/manga/manga_inpaintor.jit"
)
MANGA_LINE_MODEL_URL = os.environ.get(
    "MANGA_LINE_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/manga/erika.jit"
)


class Manga(InpaintModel):
    pad_mod = 16

    def init_model(self, device, **kwargs):
        self.inpaintor_model = load_jit_model(MANGA_INPAINTOR_MODEL_URL, device)
        self.line_model = load_jit_model(MANGA_LINE_MODEL_URL, device)
        self.seed = 42

    @staticmethod
    def is_downloaded() -> bool:
        model_paths = [
            get_cache_path_by_url(MANGA_INPAINTOR_MODEL_URL),
            get_cache_path_by_url(MANGA_LINE_MODEL_URL),
        ]
        return all([os.path.exists(it) for it in model_paths])

    def forward(self, image, mask, config: Config):
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
        gray_img = torch.from_numpy(gray_img[np.newaxis, np.newaxis, :, :].astype(np.float32)).to(self.device)
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
