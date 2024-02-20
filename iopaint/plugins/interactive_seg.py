import hashlib
from typing import List

import numpy as np
import torch
from loguru import logger

from iopaint.helper import download_model
from iopaint.plugins.base_plugin import BasePlugin
from iopaint.plugins.segment_anything import SamPredictor, sam_model_registry
from iopaint.plugins.segment_anything.predictor_hq import SamHQPredictor
from iopaint.schema import RunPluginRequest

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
    "mobile_sam": {
        "url": "https://github.com/Sanster/models/releases/download/MobileSAM/mobile_sam.pt",
        "md5": "f3c0d8cda613564d499310dab6c812cd",
    },
    "sam_hq_vit_b": {
        "url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth",
        "md5": "c6b8953247bcfdc8bb8ef91e36a6cacc",
    },
    "sam_hq_vit_l": {
        "url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth",
        "md5": "08947267966e4264fb39523eccc33f86",
    },
    "sam_hq_vit_h": {
        "url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
        "md5": "3560f6b6a5a6edacd814a1325c39640a",
    },
}


class InteractiveSeg(BasePlugin):
    name = "InteractiveSeg"
    support_gen_mask = True

    def __init__(self, model_name, device):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._init_session(model_name)

    def _init_session(self, model_name: str):
        model_path = download_model(
            SEGMENT_ANYTHING_MODELS[model_name]["url"],
            SEGMENT_ANYTHING_MODELS[model_name]["md5"],
        )
        logger.info(f"SegmentAnything model path: {model_path}")
        if "sam_hq" in model_name:
            self.predictor = SamHQPredictor(
                sam_model_registry[model_name](checkpoint=model_path).to(self.device)
            )
        else:
            self.predictor = SamPredictor(
                sam_model_registry[model_name](checkpoint=model_path).to(self.device)
            )
        self.prev_img_md5 = None

    def switch_model(self, new_model_name):
        if self.model_name == new_model_name:
            return

        logger.info(
            f"Switching InteractiveSeg model from {self.model_name} to {new_model_name}"
        )
        self._init_session(new_model_name)
        self.model_name = new_model_name

    def gen_mask(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        img_md5 = hashlib.md5(req.image.encode("utf-8")).hexdigest()
        return self.forward(rgb_np_img, req.clicks, img_md5)

    @torch.inference_mode()
    def forward(self, rgb_np_img, clicks: List[List], img_md5: str):
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
        return mask
