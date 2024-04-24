import abc
from typing import Optional

import cv2
import torch
import numpy as np
from loguru import logger

from iopaint.helper import (
    boxes_from_mask,
    resize_max_size,
    pad_img_to_modulo,
    switch_mps_device,
)
from iopaint.schema import InpaintRequest, HDStrategy, SDSampler
from .helper.g_diffuser_bot import expand_image
from .utils import get_scheduler


class InpaintModel:
    name = "base"
    min_size: Optional[int] = None
    pad_mod = 8
    pad_to_square = False
    is_erase_model = False

    def __init__(self, device, **kwargs):
        """

        Args:
            device:
        """
        device = switch_mps_device(self.name, device)
        self.device = device
        self.init_model(device, **kwargs)

    @abc.abstractmethod
    def init_model(self, device, **kwargs): ...

    @staticmethod
    @abc.abstractmethod
    def is_downloaded() -> bool:
        return False

    @abc.abstractmethod
    def forward(self, image, mask, config: InpaintRequest):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W, 1] 255 为 masks 区域
        return: BGR IMAGE
        """
        ...

    @staticmethod
    def download(): ...

    def _pad_forward(self, image, mask, config: InpaintRequest):
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(
            image, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )
        pad_mask = pad_img_to_modulo(
            mask, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )

        # logger.info(f"final forward pad size: {pad_image.shape}")

        image, mask = self.forward_pre_process(image, mask, config)

        result = self.forward(pad_image, pad_mask, config)
        result = result[0:origin_height, 0:origin_width, :]

        result, image, mask = self.forward_post_process(result, image, mask, config)

        if config.sd_keep_unmasked_area:
            mask = mask[:, :, np.newaxis]
            result = result * (mask / 255) + image[:, :, ::-1] * (1 - (mask / 255))
        return result

    def forward_pre_process(self, image, mask, config):
        return image, mask

    def forward_post_process(self, result, image, mask, config):
        return result, image, mask

    @torch.no_grad()
    def __call__(self, image, mask, config: InpaintRequest):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        inpaint_result = None
        # logger.info(f"hd_strategy: {config.hd_strategy}")
        if config.hd_strategy == HDStrategy.CROP:
            if max(image.shape) > config.hd_strategy_crop_trigger_size:
                logger.info("Run crop strategy")
                boxes = boxes_from_mask(mask)
                crop_result = []
                for box in boxes:
                    crop_image, crop_box = self._run_box(image, mask, box, config)
                    crop_result.append((crop_image, crop_box))

                inpaint_result = image[:, :, ::-1]
                for crop_image, crop_box in crop_result:
                    x1, y1, x2, y2 = crop_box
                    inpaint_result[y1:y2, x1:x2, :] = crop_image

        elif config.hd_strategy == HDStrategy.RESIZE:
            if max(image.shape) > config.hd_strategy_resize_limit:
                origin_size = image.shape[:2]
                downsize_image = resize_max_size(
                    image, size_limit=config.hd_strategy_resize_limit
                )
                downsize_mask = resize_max_size(
                    mask, size_limit=config.hd_strategy_resize_limit
                )

                logger.info(
                    f"Run resize strategy, origin size: {image.shape} forward size: {downsize_image.shape}"
                )
                inpaint_result = self._pad_forward(
                    downsize_image, downsize_mask, config
                )

                # only paste masked area result
                inpaint_result = cv2.resize(
                    inpaint_result,
                    (origin_size[1], origin_size[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
                original_pixel_indices = mask < 127
                inpaint_result[original_pixel_indices] = image[:, :, ::-1][
                    original_pixel_indices
                ]

        if inpaint_result is None:
            inpaint_result = self._pad_forward(image, mask, config)

        return inpaint_result

    def _crop_box(self, image, mask, box, config: InpaintRequest):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE, (l, r, r, b)
        """
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        img_h, img_w = image.shape[:2]

        w = box_w + config.hd_strategy_crop_margin * 2
        h = box_h + config.hd_strategy_crop_margin * 2

        _l = cx - w // 2
        _r = cx + w // 2
        _t = cy - h // 2
        _b = cy + h // 2

        l = max(_l, 0)
        r = min(_r, img_w)
        t = max(_t, 0)
        b = min(_b, img_h)

        # try to get more context when crop around image edge
        if _l < 0:
            r += abs(_l)
        if _r > img_w:
            l -= _r - img_w
        if _t < 0:
            b += abs(_t)
        if _b > img_h:
            t -= _b - img_h

        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]

        # logger.info(f"box size: ({box_h},{box_w}) crop size: {crop_img.shape}")

        return crop_img, crop_mask, [l, t, r, b]

    def _calculate_cdf(self, histogram):
        cdf = histogram.cumsum()
        normalized_cdf = cdf / float(cdf.max())
        return normalized_cdf

    def _calculate_lookup(self, source_cdf, reference_cdf):
        lookup_table = np.zeros(256)
        lookup_val = 0
        for source_index, source_val in enumerate(source_cdf):
            for reference_index, reference_val in enumerate(reference_cdf):
                if reference_val >= source_val:
                    lookup_val = reference_index
                    break
            lookup_table[source_index] = lookup_val
        return lookup_table

    def _match_histograms(self, source, reference, mask):
        transformed_channels = []
        if len(mask.shape) == 3:
            mask = mask[:, :, -1]

        for channel in range(source.shape[-1]):
            source_channel = source[:, :, channel]
            reference_channel = reference[:, :, channel]

            # only calculate histograms for non-masked parts
            source_histogram, _ = np.histogram(source_channel[mask == 0], 256, [0, 256])
            reference_histogram, _ = np.histogram(
                reference_channel[mask == 0], 256, [0, 256]
            )

            source_cdf = self._calculate_cdf(source_histogram)
            reference_cdf = self._calculate_cdf(reference_histogram)

            lookup = self._calculate_lookup(source_cdf, reference_cdf)

            transformed_channels.append(cv2.LUT(source_channel, lookup))

        result = cv2.merge(transformed_channels)
        result = cv2.convertScaleAbs(result)

        return result

    def _apply_cropper(self, image, mask, config: InpaintRequest):
        img_h, img_w = image.shape[:2]
        l, t, w, h = (
            config.croper_x,
            config.croper_y,
            config.croper_width,
            config.croper_height,
        )
        r = l + w
        b = t + h

        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]
        return crop_img, crop_mask, (l, t, r, b)

    def _run_box(self, image, mask, box, config: InpaintRequest):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE
        """
        crop_img, crop_mask, [l, t, r, b] = self._crop_box(image, mask, box, config)

        return self._pad_forward(crop_img, crop_mask, config), [l, t, r, b]


class DiffusionInpaintModel(InpaintModel):
    def __init__(self, device, **kwargs):
        self.model_info = kwargs["model_info"]
        self.model_id_or_path = self.model_info.path
        super().__init__(device, **kwargs)

    @torch.no_grad()
    def __call__(self, image, mask, config: InpaintRequest):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        # boxes = boxes_from_mask(mask)
        if config.use_croper:
            crop_img, crop_mask, (l, t, r, b) = self._apply_cropper(image, mask, config)
            crop_image = self._scaled_pad_forward(crop_img, crop_mask, config)
            inpaint_result = image[:, :, ::-1]
            inpaint_result[t:b, l:r, :] = crop_image
        elif config.use_extender:
            inpaint_result = self._do_outpainting(image, config)
        else:
            inpaint_result = self._scaled_pad_forward(image, mask, config)

        return inpaint_result

    def _do_outpainting(self, image, config: InpaintRequest):
        # cropper 和 image 在同一个坐标系下，croper_x/y 可能为负数
        # 从 image 中 crop 出 outpainting 区域
        image_h, image_w = image.shape[:2]
        cropper_l = config.extender_x
        cropper_t = config.extender_y
        cropper_r = config.extender_x + config.extender_width
        cropper_b = config.extender_y + config.extender_height
        image_l = 0
        image_t = 0
        image_r = image_w
        image_b = image_h

        # 类似求 IOU
        l = max(cropper_l, image_l)
        t = max(cropper_t, image_t)
        r = min(cropper_r, image_r)
        b = min(cropper_b, image_b)

        assert (
            0 <= l < r and 0 <= t < b
        ), f"cropper and image not overlap, {l},{t},{r},{b}"

        cropped_image = image[t:b, l:r, :]
        padding_l = max(0, image_l - cropper_l)
        padding_t = max(0, image_t - cropper_t)
        padding_r = max(0, cropper_r - image_r)
        padding_b = max(0, cropper_b - image_b)

        expanded_image, mask_image = expand_image(
            cropped_image,
            left=padding_l,
            top=padding_t,
            right=padding_r,
            bottom=padding_b,
        )

        # 最终扩大了的 image, BGR
        expanded_cropped_result_image = self._scaled_pad_forward(
            expanded_image, mask_image, config
        )

        # RGB -> BGR
        outpainting_image = cv2.copyMakeBorder(
            image,
            left=padding_l,
            top=padding_t,
            right=padding_r,
            bottom=padding_b,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )[:, :, ::-1]

        # 把 cropped_result_image 贴到 outpainting_image 上，这一步不需要 blend
        paste_t = 0 if config.extender_y < 0 else config.extender_y
        paste_l = 0 if config.extender_x < 0 else config.extender_x

        outpainting_image[
            paste_t : paste_t + expanded_cropped_result_image.shape[0],
            paste_l : paste_l + expanded_cropped_result_image.shape[1],
            :,
        ] = expanded_cropped_result_image
        return outpainting_image

    def _scaled_pad_forward(self, image, mask, config: InpaintRequest):
        longer_side_length = int(config.sd_scale * max(image.shape[:2]))
        origin_size = image.shape[:2]
        downsize_image = resize_max_size(image, size_limit=longer_side_length)
        downsize_mask = resize_max_size(mask, size_limit=longer_side_length)
        if config.sd_scale != 1:
            logger.info(
                f"Resize image to do sd inpainting: {image.shape} -> {downsize_image.shape}"
            )
        inpaint_result = self._pad_forward(downsize_image, downsize_mask, config)
        # only paste masked area result
        inpaint_result = cv2.resize(
            inpaint_result,
            (origin_size[1], origin_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        return inpaint_result

    def set_scheduler(self, config: InpaintRequest):
        scheduler_config = self.model.scheduler.config
        sd_sampler = config.sd_sampler
        if config.sd_lcm_lora and self.model_info.support_lcm_lora:
            sd_sampler = SDSampler.lcm
            logger.info(f"LCM Lora enabled, use {sd_sampler} sampler")
        scheduler = get_scheduler(sd_sampler, scheduler_config)
        self.model.scheduler = scheduler

    def forward_pre_process(self, image, mask, config):
        if config.sd_mask_blur != 0:
            k = 2 * config.sd_mask_blur + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        return image, mask

    def forward_post_process(self, result, image, mask, config):
        if config.sd_match_histograms:
            result = self._match_histograms(result, image[:, :, ::-1], mask)

        if config.use_extender and config.sd_mask_blur != 0:
            k = 2 * config.sd_mask_blur + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return result, image, mask
