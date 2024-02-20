import base64
import imghdr
import io
import os
import sys
from typing import List, Optional, Dict, Tuple

from urllib.parse import urlparse
import cv2
from PIL import Image, ImageOps, PngImagePlugin
import numpy as np
import torch
from iopaint.const import MPS_UNSUPPORT_MODELS
from loguru import logger
from torch.hub import download_url_to_file, get_dir
import hashlib


def md5sum(filename):
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def switch_mps_device(model_name, device):
    if model_name in MPS_UNSUPPORT_MODELS and str(device) == "mps":
        logger.info(f"{model_name} not support mps, switch to cpu")
        return torch.device("cpu")
    return device


def get_cache_path_by_url(url):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def download_model(url, model_md5: str = None):
    if os.path.exists(url):
        cached_file = url
    else:
        cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
        if model_md5:
            _md5 = md5sum(cached_file)
            if model_md5 == _md5:
                logger.info(f"Download model success, md5: {_md5}")
            else:
                try:
                    os.remove(cached_file)
                    logger.error(
                        f"Model md5: {_md5}, expected md5: {model_md5}, wrong model deleted. Please restart iopaint."
                        f"If you still have errors, please try download model manually first https://lama-cleaner-docs.vercel.app/install/download_model_manually.\n"
                    )
                except:
                    logger.error(
                        f"Model md5: {_md5}, expected md5: {model_md5}, please delete {cached_file} and restart iopaint."
                    )
                exit(-1)

    return cached_file


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def handle_error(model_path, model_md5, e):
    _md5 = md5sum(model_path)
    if _md5 != model_md5:
        try:
            os.remove(model_path)
            logger.error(
                f"Model md5: {_md5}, expected md5: {model_md5}, wrong model deleted. Please restart iopaint."
                f"If you still have errors, please try download model manually first https://lama-cleaner-docs.vercel.app/install/download_model_manually.\n"
            )
        except:
            logger.error(
                f"Model md5: {_md5}, expected md5: {model_md5}, please delete {model_path} and restart iopaint."
            )
    else:
        logger.error(
            f"Failed to load model {model_path},"
            f"please submit an issue at https://github.com/Sanster/lama-cleaner/issues and include a screenshot of the error:\n{e}"
        )
    exit(-1)


def load_jit_model(url_or_path, device, model_md5: str):
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path, model_md5)

    logger.info(f"Loading model from: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location="cpu").to(device)
    except Exception as e:
        handle_error(model_path, model_md5, e)
    model.eval()
    return model


def load_model(model: torch.nn.Module, url_or_path, device, model_md5):
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path, model_md5)

    try:
        logger.info(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
    except Exception as e:
        handle_error(model_path, model_md5, e)
    model.eval()
    return model


def numpy_to_bytes(image_numpy: np.ndarray, ext: str) -> bytes:
    data = cv2.imencode(
        f".{ext}",
        image_numpy,
        [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(cv2.IMWRITE_PNG_COMPRESSION), 0],
    )[1]
    image_bytes = data.tobytes()
    return image_bytes


def pil_to_bytes(pil_img, ext: str, quality: int = 95, infos={}) -> bytes:
    with io.BytesIO() as output:
        kwargs = {k: v for k, v in infos.items() if v is not None}
        if ext == "jpg":
            ext = "jpeg"
        if "png" == ext.lower() and "parameters" in kwargs:
            pnginfo_data = PngImagePlugin.PngInfo()
            pnginfo_data.add_text("parameters", kwargs["parameters"])
            kwargs["pnginfo"] = pnginfo_data

        pil_img.save(output, format=ext, quality=quality, **kwargs)
        image_bytes = output.getvalue()
    return image_bytes


def load_img(img_bytes, gray: bool = False, return_info: bool = False):
    alpha_channel = None
    image = Image.open(io.BytesIO(img_bytes))

    if return_info:
        infos = image.info

    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass

    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)

    if return_info:
        return np_img, alpha_channel, infos
    return np_img, alpha_channel


def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


def resize_max_size(
    np_img, size_limit: int, interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    # Resize image's longer size to size_limit if longer size larger than size_limit
    h, w = np_img.shape[:2]
    if max(h, w) > size_limit:
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)
        new_h = int(h * ratio + 0.5)
        return cv2.resize(np_img, dsize=(new_w, new_h), interpolation=interpolation)
    else:
        return np_img


def pad_img_to_modulo(
    img: np.ndarray, mod: int, square: bool = False, min_size: Optional[int] = None
):
    """

    Args:
        img: [H, W, C]
        mod:
        square: 是否为正方形
        min_size:

    Returns:

    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def boxes_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        mask: (h, w, 1)  0~255

    Returns:

    """
    height, width = mask.shape[:2]
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([x, y, x + w, y + h]).astype(int)

        box[::2] = np.clip(box[::2], 0, width)
        box[1::2] = np.clip(box[1::2], 0, height)
        boxes.append(box)

    return boxes


def only_keep_largest_contour(mask: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        mask: (h, w)  0~255

    Returns:

    """
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_index = -1
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_index = i

    if max_index != -1:
        new_mask = np.zeros_like(mask)
        return cv2.drawContours(new_mask, contours, max_index, 255, -1)
    else:
        return mask


def is_mac():
    return sys.platform == "darwin"


def get_image_ext(img_bytes):
    w = imghdr.what("", img_bytes)
    if w is None:
        w = "jpeg"
    return w


def decode_base64_to_image(
    encoding: str, gray=False
) -> Tuple[np.array, Optional[np.array], Dict]:
    if encoding.startswith("data:image/") or encoding.startswith(
        "data:application/octet-stream;base64,"
    ):
        encoding = encoding.split(";")[1].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoding)))

    alpha_channel = None
    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass
    # exif_transpose will remove exif rotate info，we must call image.info after exif_transpose
    infos = image.info

    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)

    return np_img, alpha_channel, infos


def encode_pil_to_base64(image: Image, quality: int, infos: Dict) -> bytes:
    img_bytes = pil_to_bytes(
        image,
        "png",
        quality=quality,
        infos=infos,
    )
    return base64.b64encode(img_bytes)


def concat_alpha_channel(rgb_np_img, alpha_channel) -> np.ndarray:
    if alpha_channel is not None:
        if alpha_channel.shape[:2] != rgb_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(rgb_np_img.shape[1], rgb_np_img.shape[0])
            )
        rgb_np_img = np.concatenate(
            (rgb_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )
    return rgb_np_img


def adjust_mask(mask: np.ndarray, kernel_size: int, operate):
    # fronted brush color "ffcc00bb"
    # kernel_size = kernel_size*2+1
    mask[mask >= 127] = 255
    mask[mask < 127] = 0

    if operate == "reverse":
        mask = 255 - mask
    else:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1)
        )
        if operate == "expand":
            mask = cv2.dilate(
                mask,
                kernel,
                iterations=1,
            )
        else:
            mask = cv2.erode(
                mask,
                kernel,
                iterations=1,
            )
    res_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    res_mask[mask > 128] = [255, 203, 0, int(255 * 0.73)]
    res_mask = cv2.cvtColor(res_mask, cv2.COLOR_BGRA2RGBA)
    return res_mask


def gen_frontend_mask(bgr_or_gray_mask):
    if len(bgr_or_gray_mask.shape) == 3 and bgr_or_gray_mask.shape[2] != 1:
        bgr_or_gray_mask = cv2.cvtColor(bgr_or_gray_mask, cv2.COLOR_BGR2GRAY)

    # fronted brush color "ffcc00bb"
    # TODO: how to set kernel size?
    kernel_size = 9
    bgr_or_gray_mask = cv2.dilate(
        bgr_or_gray_mask,
        np.ones((kernel_size, kernel_size), np.uint8),
        iterations=1,
    )
    res_mask = np.zeros(
        (bgr_or_gray_mask.shape[0], bgr_or_gray_mask.shape[1], 4), dtype=np.uint8
    )
    res_mask[bgr_or_gray_mask > 128] = [255, 203, 0, int(255 * 0.73)]
    res_mask = cv2.cvtColor(res_mask, cv2.COLOR_BGRA2RGBA)
    return res_mask
