import os
from typing import List

import cv2
import torch
import numpy as np

from lama_cleaner.helper import pad_img_to_modulo, download_model, boxes_from_mask

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)


class LaMa:
    def __init__(self, crop_trigger_size: List[int], crop_size: List[int], device):
        """

        Args:
            crop_trigger_size: h, w
            crop_size: h, w
            device:
        """
        self.crop_trigger_size = crop_trigger_size
        self.crop_size = crop_size
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
        area = image.shape[1] * image.shape[2]
        if area < self.crop_trigger_size[0] * self.crop_trigger_size[1]:
            return self._run(image, mask)

        print("Trigger crop image")
        boxes = boxes_from_mask(mask)
        crop_result = []
        for box in boxes:
            crop_image, crop_box = self._run_box(image, mask, box)
            crop_result.append((crop_image, crop_box))

        image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, ::-1]
        for crop_image, crop_box in crop_result:
            x1, y1, x2, y2 = crop_box
            image[y1:y2, x1:x2, :] = crop_image
        return image

    def _run_box(self, image, mask, box):
        """

        Args:
            image: [C, H, W] RGB
            mask: [1, H, W]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE
        """
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        crop_h, crop_w = self.crop_size
        img_h, img_w = image.shape[1:]

        # TODO: when box_w > crop_w, add some margin around?
        w = max(crop_w, box_w)
        h = max(crop_h, box_h)

        l = max(cx - w // 2, 0)
        t = max(cy - h // 2, 0)
        r = min(cx + w // 2, img_w)
        b = min(cy + h // 2, img_h)

        crop_img = image[:, t:b, l:r]
        crop_mask = mask[:, t:b, l:r]

        print(f"Apply zoom in size width x height: {crop_img.shape}")

        return self._run(crop_img, crop_mask), [l, t, r, b]

    def _run(self, image, mask):
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
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return cur_res
