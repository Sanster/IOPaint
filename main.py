#!/usr/bin/env python3

import argparse
import io
import multiprocessing
import os
import time
from typing import Union

import cv2
import torch

from lama_cleaner.lama import LaMa
from lama_cleaner.ldm import LDM

from flaskwebgui import FlaskUI

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
except:
    pass

from flask import Flask, request, send_file
from flask_cors import CORS

from lama_cleaner.helper import (
    load_img,
    norm_img,
    numpy_to_bytes,
    resize_max_size,
)

NUM_THREADS = str(multiprocessing.cpu_count())

os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
if os.environ.get("CACHE_DIR"):
    os.environ["TORCH_HOME"] = os.environ["CACHE_DIR"]

BUILD_DIR = os.environ.get("LAMA_CLEANER_BUILD_DIR",
                           "./lama_cleaner/app/build")

app = Flask(__name__, static_folder=os.path.join(BUILD_DIR, "static"))
app.config["JSON_AS_ASCII"] = False
CORS(app)

model = None
device = None


@app.route("/inpaint", methods=["POST"])
def process():
    input = request.files
    # RGB
    image = load_img(input["image"].read())
    original_shape = image.shape
    interpolation = cv2.INTER_CUBIC

    size_limit: Union[int, str] = request.form.get("sizeLimit", "1080")
    if size_limit == "Original":
        size_limit = max(image.shape)
    else:
        size_limit = int(size_limit)

    print(f"Origin image shape: {original_shape}")
    image = resize_max_size(image, size_limit=size_limit,
                            interpolation=interpolation)
    print(f"Resized image shape: {image.shape}")
    image = norm_img(image)

    mask = load_img(input["mask"].read(), gray=True)
    mask = resize_max_size(mask, size_limit=size_limit,
                           interpolation=interpolation)
    mask = norm_img(mask)

    start = time.time()
    res_np_img = model(image, mask)
    print(f"process time: {(time.time() - start) * 1000}ms")

    torch.cuda.empty_cache()

    return send_file(
        io.BytesIO(numpy_to_bytes(res_np_img)),
        mimetype="image/jpeg",
        as_attachment=True,
        attachment_filename="result.jpeg",
    )


@app.route("/")
def index():
    return send_file(os.path.join(BUILD_DIR, "index.html"))


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--model", default="lama", choices=["lama", "ldm"])
    parser.add_argument("--crop-trigger-size", default="2042,2042",
                        help="If image size large then crop-trigger-size, "
                             "crop each area from original image to do inference."
                             "Mainly for performance and memory reasons"
                             "Only for lama")
    parser.add_argument("--crop-size", default="512,512")
    parser.add_argument(
        "--ldm-steps",
        default=50,
        type=int,
        help="Steps for DDIM sampling process."
             "The larger the value, the better the result, but it will be more time-consuming",
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--gui", action="store_true",
                        help="Launch as desktop app")
    parser.add_argument("--gui_size", default=[1600, 1000], nargs=2, type=int,
                        help="Set window size for GUI")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    global model
    global device
    args = get_args_parser()
    device = torch.device(args.device)

    crop_trigger_size = [int(it) for it in args.crop_trigger_size.split(",")]
    crop_size = [int(it) for it in args.crop_size.split(",")]

    if args.model == "lama":
        model = LaMa(crop_trigger_size=crop_trigger_size,
                     crop_size=crop_size, device=device)
    elif args.model == "ldm":
        model = LDM(device, steps=args.ldm_steps)
    else:
        raise NotImplementedError(f"Not supported model: {args.model}")

    if args.gui:
        app_width, app_height = args.gui_size
        ui = FlaskUI(app, width=app_width, height=app_height)
        ui.run()
    else:
        app.run(host="127.0.0.1", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
