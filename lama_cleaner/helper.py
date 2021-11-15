import os
import sys

from urllib.parse import urlparse
import cv2
import numpy as np
import torch
from torch.hub import download_url_to_file, get_dir

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)


def download_model(url=LAMA_MODEL_URL):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    filename = os.path.basename(parts.path)

    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
    return cached_file


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def numpy_to_bytes(image_numpy: np.ndarray) -> bytes:
    data = cv2.imencode(".jpg", image_numpy)[1]
    image_bytes = data.tobytes()
    return image_bytes


def load_img(img_bytes, gray: bool = False, norm: bool = True):
    nparr = np.frombuffer(img_bytes, np.uint8)
    if gray:
        np_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
    else:
        np_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    if norm:
        np_img = np.transpose(np_img, (2, 0, 1))
        np_img = np_img.astype("float32") / 255

    return np_img


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )
