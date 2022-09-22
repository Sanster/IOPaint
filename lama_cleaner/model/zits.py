import os
import time

import cv2
import skimage
import torch
import torch.nn.functional as F

from lama_cleaner.helper import get_cache_path_by_url, load_jit_model
from lama_cleaner.schema import Config
import numpy as np

from lama_cleaner.model.base import InpaintModel

ZITS_INPAINT_MODEL_URL = os.environ.get(
    "ZITS_INPAINT_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_zits/zits-inpaint-0717.pt",
)

ZITS_EDGE_LINE_MODEL_URL = os.environ.get(
    "ZITS_EDGE_LINE_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_zits/zits-edge-line-0717.pt",
)

ZITS_STRUCTURE_UPSAMPLE_MODEL_URL = os.environ.get(
    "ZITS_STRUCTURE_UPSAMPLE_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_zits/zits-structure-upsample-0717.pt",
)

ZITS_WIRE_FRAME_MODEL_URL = os.environ.get(
    "ZITS_WIRE_FRAME_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_zits/zits-wireframe-0717.pt",
)


def resize(img, height, width, center_crop=False):
    imgh, imgw = img.shape[0:2]

    if center_crop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j : j + side, i : i + side, ...]

    if imgh > height and imgw > width:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_LINEAR
    img = cv2.resize(img, (height, width), interpolation=inter)

    return img


def to_tensor(img, scale=True, norm=False):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    c = img.shape[-1]

    if scale:
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().div(255)
    else:
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()

    if norm:
        mean = torch.tensor([0.5, 0.5, 0.5]).reshape(c, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).reshape(c, 1, 1)
        img_t = (img_t - mean) / std
    return img_t


def load_masked_position_encoding(mask):
    ones_filter = np.ones((3, 3), dtype=np.float32)
    d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
    d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
    d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
    d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
    str_size = 256
    pos_num = 128

    ori_mask = mask.copy()
    ori_h, ori_w = ori_mask.shape[0:2]
    ori_mask = ori_mask / 255
    mask = cv2.resize(mask, (str_size, str_size), interpolation=cv2.INTER_AREA)
    mask[mask > 0] = 255
    h, w = mask.shape[0:2]
    mask3 = mask.copy()
    mask3 = 1.0 - (mask3 / 255.0)
    pos = np.zeros((h, w), dtype=np.int32)
    direct = np.zeros((h, w, 4), dtype=np.int32)
    i = 0
    while np.sum(1 - mask3) > 0:
        i += 1
        mask3_ = cv2.filter2D(mask3, -1, ones_filter)
        mask3_[mask3_ > 0] = 1
        sub_mask = mask3_ - mask3
        pos[sub_mask == 1] = i

        m = cv2.filter2D(mask3, -1, d_filter1)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 0] = 1

        m = cv2.filter2D(mask3, -1, d_filter2)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 1] = 1

        m = cv2.filter2D(mask3, -1, d_filter3)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 2] = 1

        m = cv2.filter2D(mask3, -1, d_filter4)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 3] = 1

        mask3 = mask3_

    abs_pos = pos.copy()
    rel_pos = pos / (str_size / 2)  # to 0~1 maybe larger than 1
    rel_pos = (rel_pos * pos_num).astype(np.int32)
    rel_pos = np.clip(rel_pos, 0, pos_num - 1)

    if ori_w != w or ori_h != h:
        rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        rel_pos[ori_mask == 0] = 0
        direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        direct[ori_mask == 0, :] = 0

    return rel_pos, abs_pos, direct


def load_image(img, mask, device, sigma256=3.0):
    """
    Args:
        img: [H, W, C] RGB
        mask: [H, W] 255 为 masks 区域
        sigma256:

    Returns:

    """
    h, w, _ = img.shape
    imgh, imgw = img.shape[0:2]
    img_256 = resize(img, 256, 256)

    mask = (mask > 127).astype(np.uint8) * 255
    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
    mask_256[mask_256 > 0] = 255

    mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_AREA)
    mask_512[mask_512 > 0] = 255

    # original skimage implemention
    # https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny
    # low_threshold: Lower bound for hysteresis thresholding (linking edges). If None, low_threshold is set to 10% of dtype’s max.
    # high_threshold: Upper bound for hysteresis thresholding (linking edges). If None, high_threshold is set to 20% of dtype’s max.
    gray_256 = skimage.color.rgb2gray(img_256)
    edge_256 = skimage.feature.canny(gray_256, sigma=sigma256, mask=None).astype(float)
    # cv2.imwrite("skimage_gray.jpg", (_gray_256*255).astype(np.uint8))
    # cv2.imwrite("skimage_edge.jpg", (_edge_256*255).astype(np.uint8))

    # gray_256 = cv2.cvtColor(img_256, cv2.COLOR_RGB2GRAY)
    # gray_256_blured = cv2.GaussianBlur(gray_256, ksize=(3,3), sigmaX=sigma256, sigmaY=sigma256)
    # edge_256 = cv2.Canny(gray_256_blured, threshold1=int(255*0.1), threshold2=int(255*0.2))
    # cv2.imwrite("edge.jpg", edge_256)

    # line
    img_512 = resize(img, 512, 512)

    rel_pos, abs_pos, direct = load_masked_position_encoding(mask)

    batch = dict()
    batch["images"] = to_tensor(img.copy()).unsqueeze(0).to(device)
    batch["img_256"] = to_tensor(img_256, norm=True).unsqueeze(0).to(device)
    batch["masks"] = to_tensor(mask).unsqueeze(0).to(device)
    batch["mask_256"] = to_tensor(mask_256).unsqueeze(0).to(device)
    batch["mask_512"] = to_tensor(mask_512).unsqueeze(0).to(device)
    batch["edge_256"] = to_tensor(edge_256, scale=False).unsqueeze(0).to(device)
    batch["img_512"] = to_tensor(img_512).unsqueeze(0).to(device)
    batch["rel_pos"] = torch.LongTensor(rel_pos).unsqueeze(0).to(device)
    batch["abs_pos"] = torch.LongTensor(abs_pos).unsqueeze(0).to(device)
    batch["direct"] = torch.LongTensor(direct).unsqueeze(0).to(device)
    batch["h"] = imgh
    batch["w"] = imgw

    return batch


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    if isinstance(data, list):
        return [to_device(d, device) for d in data]


class ZITS(InpaintModel):
    min_size = 256
    pad_mod = 32
    pad_to_square = True

    def __init__(self, device, **kwargs):
        """

        Args:
            device:
        """
        super().__init__(device)
        self.device = device
        self.sample_edge_line_iterations = 1

    def init_model(self, device, **kwargs):
        self.wireframe = load_jit_model(ZITS_WIRE_FRAME_MODEL_URL, device)
        self.edge_line = load_jit_model(ZITS_EDGE_LINE_MODEL_URL, device)
        self.structure_upsample = load_jit_model(
            ZITS_STRUCTURE_UPSAMPLE_MODEL_URL, device
        )
        self.inpaint = load_jit_model(ZITS_INPAINT_MODEL_URL, device)

    @staticmethod
    def is_downloaded() -> bool:
        model_paths = [
            get_cache_path_by_url(ZITS_WIRE_FRAME_MODEL_URL),
            get_cache_path_by_url(ZITS_EDGE_LINE_MODEL_URL),
            get_cache_path_by_url(ZITS_STRUCTURE_UPSAMPLE_MODEL_URL),
            get_cache_path_by_url(ZITS_INPAINT_MODEL_URL),
        ]
        return all([os.path.exists(it) for it in model_paths])

    def wireframe_edge_and_line(self, items, enable: bool):
        # 最终向 items 中添加 edge 和 line key
        if not enable:
            items["edge"] = torch.zeros_like(items["masks"])
            items["line"] = torch.zeros_like(items["masks"])
            return

        start = time.time()
        try:
            line_256 = self.wireframe_forward(
                items["img_512"],
                h=256,
                w=256,
                masks=items["mask_512"],
                mask_th=0.85,
            )
        except:
            line_256 = torch.zeros_like(items["mask_256"])

        print(f"wireframe_forward time: {(time.time() - start) * 1000:.2f}ms")

        # np_line = (line[0][0].numpy() * 255).astype(np.uint8)
        # cv2.imwrite("line.jpg", np_line)

        start = time.time()
        edge_pred, line_pred = self.sample_edge_line_logits(
            context=[items["img_256"], items["edge_256"], line_256],
            mask=items["mask_256"].clone(),
            iterations=self.sample_edge_line_iterations,
            add_v=0.05,
            mul_v=4,
        )
        print(f"sample_edge_line_logits time: {(time.time() - start) * 1000:.2f}ms")

        # np_edge_pred = (edge_pred[0][0].numpy() * 255).astype(np.uint8)
        # cv2.imwrite("edge_pred.jpg", np_edge_pred)
        # np_line_pred = (line_pred[0][0].numpy() * 255).astype(np.uint8)
        # cv2.imwrite("line_pred.jpg", np_line_pred)
        # exit()

        input_size = min(items["h"], items["w"])
        if input_size != 256 and input_size > 256:
            while edge_pred.shape[2] < input_size:
                edge_pred = self.structure_upsample(edge_pred)
                edge_pred = torch.sigmoid((edge_pred + 2) * 2)

                line_pred = self.structure_upsample(line_pred)
                line_pred = torch.sigmoid((line_pred + 2) * 2)

            edge_pred = F.interpolate(
                edge_pred,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
            line_pred = F.interpolate(
                line_pred,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )

        # np_edge_pred = (edge_pred[0][0].numpy() * 255).astype(np.uint8)
        # cv2.imwrite("edge_pred_upsample.jpg", np_edge_pred)
        # np_line_pred = (line_pred[0][0].numpy() * 255).astype(np.uint8)
        # cv2.imwrite("line_pred_upsample.jpg", np_line_pred)
        # exit()

        items["edge"] = edge_pred.detach()
        items["line"] = line_pred.detach()

    @torch.no_grad()
    def forward(self, image, mask, config: Config):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W]
        return: BGR IMAGE
        """
        mask = mask[:, :, 0]
        items = load_image(image, mask, device=self.device)

        self.wireframe_edge_and_line(items, config.zits_wireframe)

        inpainted_image = self.inpaint(
            items["images"],
            items["masks"],
            items["edge"],
            items["line"],
            items["rel_pos"],
            items["direct"],
        )

        inpainted_image = inpainted_image * 255.0
        inpainted_image = (
            inpainted_image.cpu().permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
        )
        inpainted_image = inpainted_image[:, :, ::-1]

        # cv2.imwrite("inpainted.jpg", inpainted_image)
        # exit()

        return inpainted_image

    def wireframe_forward(self, images, h, w, masks, mask_th=0.925):
        lcnn_mean = torch.tensor([109.730, 103.832, 98.681]).reshape(1, 3, 1, 1)
        lcnn_std = torch.tensor([22.275, 22.124, 23.229]).reshape(1, 3, 1, 1)
        images = images * 255.0
        # the masks value of lcnn is 127.5
        masked_images = images * (1 - masks) + torch.ones_like(images) * masks * 127.5
        masked_images = (masked_images - lcnn_mean) / lcnn_std

        def to_int(x):
            return tuple(map(int, x))

        lines_tensor = []
        lmap = np.zeros((h, w))

        output_masked = self.wireframe(masked_images)

        output_masked = to_device(output_masked, "cpu")
        if output_masked["num_proposals"] == 0:
            lines_masked = []
            scores_masked = []
        else:
            lines_masked = output_masked["lines_pred"].numpy()
            lines_masked = [
                [line[1] * h, line[0] * w, line[3] * h, line[2] * w]
                for line in lines_masked
            ]
            scores_masked = output_masked["lines_score"].numpy()

        for line, score in zip(lines_masked, scores_masked):
            if score > mask_th:
                rr, cc, value = skimage.draw.line_aa(
                    *to_int(line[0:2]), *to_int(line[2:4])
                )
                lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

        lmap = np.clip(lmap * 255, 0, 255).astype(np.uint8)
        lines_tensor.append(to_tensor(lmap).unsqueeze(0))

        lines_tensor = torch.cat(lines_tensor, dim=0)
        return lines_tensor.detach().to(self.device)

    def sample_edge_line_logits(
        self, context, mask=None, iterations=1, add_v=0, mul_v=4
    ):
        [img, edge, line] = context

        img = img * (1 - mask)
        edge = edge * (1 - mask)
        line = line * (1 - mask)

        for i in range(iterations):
            edge_logits, line_logits = self.edge_line(img, edge, line, masks=mask)

            edge_pred = torch.sigmoid(edge_logits)
            line_pred = torch.sigmoid((line_logits + add_v) * mul_v)
            edge = edge + edge_pred * mask
            edge[edge >= 0.25] = 1
            edge[edge < 0.25] = 0
            line = line + line_pred * mask

            b, _, h, w = edge_pred.shape
            edge_pred = edge_pred.reshape(b, -1, 1)
            line_pred = line_pred.reshape(b, -1, 1)
            mask = mask.reshape(b, -1)

            edge_probs = torch.cat([1 - edge_pred, edge_pred], dim=-1)
            line_probs = torch.cat([1 - line_pred, line_pred], dim=-1)
            edge_probs[:, :, 1] += 0.5
            line_probs[:, :, 1] += 0.5
            edge_max_probs = edge_probs.max(dim=-1)[0] + (1 - mask) * (-100)
            line_max_probs = line_probs.max(dim=-1)[0] + (1 - mask) * (-100)

            indices = torch.sort(
                edge_max_probs + line_max_probs, dim=-1, descending=True
            )[1]

            for ii in range(b):
                keep = int((i + 1) / iterations * torch.sum(mask[ii, ...]))

                assert torch.sum(mask[ii][indices[ii, :keep]]) == keep, "Error!!!"
                mask[ii][indices[ii, :keep]] = 0

            mask = mask.reshape(b, 1, h, w)
            edge = edge * (1 - mask)
            line = line * (1 - mask)

        edge, line = edge.to(torch.float32), line.to(torch.float32)
        return edge, line
