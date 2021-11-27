#!/usr/bin/env python3

import io
import os
import time
import argparse
from distutils.util import strtobool
from typing import Union
import cv2
import numpy as np
import torch

from flask import Flask, request, send_file
from flask_cors import CORS
from lama_cleaner.helper import (
    download_model,
    load_img,
    norm_img,
    resize_max_size,
    numpy_to_bytes,
    pad_img_to_modulo,
)

import multiprocessing

NUM_THREADS = str(multiprocessing.cpu_count())

os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
if os.environ.get("CACHE_DIR"):
    os.environ["TORCH_HOME"] = os.environ["CACHE_DIR"]

BUILD_DIR = os.environ.get("LAMA_CLEANER_BUILD_DIR", "./lama_cleaner/app/build")

app = Flask(__name__, static_folder=os.path.join(BUILD_DIR, "static"))
app.config["JSON_AS_ASCII"] = False
CORS(app)

model = None
device = None


@app.route("/inpaint", methods=["POST"])
def process():
    input = request.files
    image = load_img(input["image"].read())
    original_shape = image.shape
    interpolation = cv2.INTER_CUBIC

    size_limit: Union[int, str] = request.form.get("sizeLimit", "1080")
    if size_limit == "Original":
        size_limit = max(image.shape)
    else:
        size_limit = int(size_limit)

    print(f"Origin image shape: {original_shape}")
    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
    print(f"Resized image shape: {image.shape}")
    image = norm_img(image)

    mask = load_img(input["mask"].read(), gray=True)
    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)
    mask = norm_img(mask)

    res_np_img = run(image, mask)

    # resize to original size
    res_np_img = cv2.resize(
        res_np_img,
        dsize=(original_shape[1], original_shape[0]),
        interpolation=interpolation,
    )

    return send_file(
        io.BytesIO(numpy_to_bytes(res_np_img)),
        mimetype="image/jpeg",
        as_attachment=True,
        attachment_filename="result.jpeg",
    )


@app.route("/")
def index():
    return send_file(os.path.join(BUILD_DIR, "index.html"))


def run(image, mask):
    """
    image: [C, H, W]
    mask: [1, H, W]
    return: BGR IMAGE
    """
    origin_height, origin_width = image.shape[1:]
    image = pad_img_to_modulo(image, mod=8)
    mask = pad_img_to_modulo(mask, mod=8)

    mask = (mask > 0) * 1
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    start = time.time()
    inpainted_image = model(image, mask)

    print(f"process time: {(time.time() - start)*1000}ms")
    cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = cur_res[0:origin_height, 0:origin_width, :]
    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_BGR2RGB)
    return cur_res


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    global model
    global device
    args = get_args_parser()
    device = torch.device(args.device)
    model_path = download_model()
    model = torch.jit.load(model_path, map_location="cpu")
    model = model.to(device)
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
