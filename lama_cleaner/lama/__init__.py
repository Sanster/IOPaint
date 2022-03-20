import os

import cv2
import torch
import numpy as np

from lama_cleaner.helper import pad_img_to_modulo, download_model

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)


class LaMa:
    def __init__(self, device):
        self.device = device

        if os.environ.get("LAMA_MODEL"):
            model_path = os.environ.get("LAMA_MODEL")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"lama torchscript model not found: {model_path}"
                )
        else:
            model_path = download_model(LAMA_MODEL_URL)

        model = torch.jit.load(model_path, map_location="cpu")
        model = model.to(device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def __call__(self, image, mask):
        """
        image: [C, H, W] RGB
        mask: [1, H, W]
        return: BGR IMAGE
        """
        device = self.device
        origin_height, origin_width = image.shape[1:]
        image = pad_img_to_modulo(image, mod=8)
        mask = pad_img_to_modulo(mask, mod=8)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = cur_res[0:origin_height, 0:origin_width, :]
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_BGR2RGB)
        return cur_res
