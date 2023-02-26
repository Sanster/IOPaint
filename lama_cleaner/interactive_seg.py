import os

import cv2
from typing import Tuple, List
import torch
import torch.nn.functional as F
from loguru import logger
from pydantic import BaseModel
import numpy as np

from lama_cleaner.helper import only_keep_largest_contour, load_jit_model


class Click(BaseModel):
    # [y, x]
    coords: Tuple[float, float]
    is_positive: bool
    indx: int

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def scale(self, x_ratio: float, y_ratio: float) -> 'Click':
        return Click(
            coords=(self.coords[0] * x_ratio, self.coords[1] * y_ratio),
            is_positive=self.is_positive,
            indx=self.indx
        )


class ResizeTrans:
    def __init__(self, size=480):
        super().__init__()
        self.crop_height = size
        self.crop_width = size

    def transform(self, image_nd, clicks_lists):
        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        image_height, image_width = image_nd.shape[2:4]
        self.image_height = image_height
        self.image_width = image_width
        image_nd_r = F.interpolate(image_nd, (self.crop_height, self.crop_width), mode='bilinear', align_corners=True)

        y_ratio = self.crop_height / image_height
        x_ratio = self.crop_width / image_width

        clicks_lists_resized = []
        for clicks_list in clicks_lists:
            clicks_list_resized = [click.scale(y_ratio, x_ratio) for click in clicks_list]
            clicks_lists_resized.append(clicks_list_resized)

        return image_nd_r, clicks_lists_resized

    def inv_transform(self, prob_map):
        new_prob_map = F.interpolate(prob_map, (self.image_height, self.image_width), mode='bilinear',
                                     align_corners=True)

        return new_prob_map


class ISPredictor(object):
    def __init__(
        self,
        model,
        device,
        open_kernel_size: int,
        dilate_kernel_size: int,
        net_clicks_limit=None,
        zoom_in=None,
        infer_size=384,
    ):
        self.model = model
        self.open_kernel_size = open_kernel_size
        self.dilate_kernel_size = dilate_kernel_size
        self.net_clicks_limit = net_clicks_limit
        self.device = device
        self.zoom_in = zoom_in
        self.infer_size = infer_size

        # self.transforms = [zoom_in] if zoom_in is not None else []

    def __call__(self, input_image: torch.Tensor, clicks: List[Click], prev_mask):
        """

        Args:
            input_image: [1, 3, H, W]  [0~1]
            clicks: List[Click]
            prev_mask: [1, 1, H, W]

        Returns:

        """
        transforms = [ResizeTrans(self.infer_size)]
        input_image = torch.cat((input_image, prev_mask), dim=1)

        # image_nd resized to infer_size
        for t in transforms:
            image_nd, clicks_lists = t.transform(input_image, [clicks])

        # image_nd.shape = [1, 4, 256, 256]
        # points_nd.sha[e = [1, 2, 3]
        # clicks_lists[0][0] Click 类
        points_nd = self.get_points_nd(clicks_lists)
        pred_logits = self.model(image_nd, points_nd)
        pred = torch.sigmoid(pred_logits)
        pred = self.post_process(pred)

        prediction = F.interpolate(pred, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

        for t in reversed(transforms):
            prediction = t.inv_transform(prediction)

        # if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
        #    return self.get_prediction(clicker)

        return prediction.cpu().numpy()[0, 0]

    def post_process(self, pred: torch.Tensor) -> torch.Tensor:
        pred_mask = pred.cpu().numpy()[0][0]
        # morph_open to remove small noise
        kernel_size = self.open_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Why dilate: make region slightly larger to avoid missing some pixels, this generally works better
        dilate_kernel_size = self.dilate_kernel_size
        if dilate_kernel_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (dilate_kernel_size, dilate_kernel_size))
            pred_mask = cv2.dilate(pred_mask, kernel, 1)
        return torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0)

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)


INTERACTIVE_SEG_MODEL_URL = os.environ.get(
    "INTERACTIVE_SEG_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/clickseg_pplnet/clickseg_pplnet.pt",
)
INTERACTIVE_SEG_MODEL_MD5 = os.environ.get("INTERACTIVE_SEG_MODEL_MD5", "8ca44b6e02bca78f62ec26a3c32376cf")


class InteractiveSeg:
    def __init__(self, infer_size=384, open_kernel_size=3, dilate_kernel_size=3):
        device = torch.device('cpu')
        model = load_jit_model(INTERACTIVE_SEG_MODEL_URL, device, INTERACTIVE_SEG_MODEL_MD5).eval()
        self.predictor = ISPredictor(model, device,
                                     infer_size=infer_size,
                                     open_kernel_size=open_kernel_size,
                                     dilate_kernel_size=dilate_kernel_size)

    def __call__(self, image, clicks, prev_mask=None):
        """

        Args:
            image: [H,W,C] RGB
            clicks:

        Returns:

        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = torch.from_numpy((image / 255).transpose(2, 0, 1)).unsqueeze(0).float()
        if prev_mask is None:
            mask = torch.zeros_like(image[:, :1, :, :])
        else:
            logger.info('InteractiveSeg run with prev_mask')
            mask = torch.from_numpy(prev_mask / 255).unsqueeze(0).unsqueeze(0).float()

        pred_probs = self.predictor(image, clicks, mask)
        pred_mask = pred_probs > 0.5
        pred_mask = (pred_mask * 255).astype(np.uint8)

        # Find largest contour
        # pred_mask = only_keep_largest_contour(pred_mask)
        # To simplify frontend process, add mask brush color here
        fg = pred_mask == 255
        bg = pred_mask != 255
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGRA)
        # frontend brush color "ffcc00bb"
        pred_mask[bg] = 0
        pred_mask[fg] = [255, 203, 0, int(255 * 0.73)]
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGRA2RGBA)
        return pred_mask
