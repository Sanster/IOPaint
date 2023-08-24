import json

import cv2
import numpy as np
from loguru import logger

from lama_cleaner.helper import download_model
from lama_cleaner.plugins.base_plugin import BasePlugin
from lama_cleaner.plugins.segment_anything import SamPredictor, sam_model_registry

# 从小到大
SEGMENT_ANYTHING_MODELS = {
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "md5": "01ec64d29a2fca3f0661936605ae66f8",
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "md5": "0b3195507c641ddb6910d2bb5adee89c",
    },
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "md5": "4b8939a88964f0f4ff5f5b2642c598a6",
    },
}


class InteractiveSeg(BasePlugin):
    name = "InteractiveSeg"

    def __init__(self, model_name, device):
        super().__init__()
        model_path = download_model(
            SEGMENT_ANYTHING_MODELS[model_name]["url"],
            SEGMENT_ANYTHING_MODELS[model_name]["md5"],
        )
        logger.info(f"SegmentAnything model path: {model_path}")
        self.predictor = SamPredictor(
            sam_model_registry[model_name](checkpoint=model_path).to(device)
        )
        self.prev_img_md5 = None

    def __call__(self, rgb_np_img, files, form):
        clicks = json.loads(form["clicks"])
        return self.forward(rgb_np_img, clicks, form["img_md5"])

    def forward(self, rgb_np_img, clicks, img_md5):
        input_point = []
        input_label = []
        for click in clicks:
            x = click[0]
            y = click[1]
            input_point.append([x, y])
            input_label.append(click[2])

        if img_md5 and img_md5 != self.prev_img_md5:
            self.prev_img_md5 = img_md5
            self.predictor.set_image(rgb_np_img)

        masks, scores, _ = self.predictor.predict(
            point_coords=np.array(input_point),
            point_labels=np.array(input_label),
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8) * 255
        # TODO: how to set kernel size?
        kernel_size = 9
        mask = cv2.dilate(
            mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1
        )
        # fronted brush color "ffcc00bb"
        res_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        res_mask[mask == 255] = [255, 203, 0, int(255 * 0.73)]
        res_mask = cv2.cvtColor(res_mask, cv2.COLOR_BGRA2RGBA)
        return res_mask
