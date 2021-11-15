#!/usr/bin/env python3

import io
import os
import time
import argparse
import cv2
import numpy as np
import torch

from flask import Flask, request, send_file
from flask_cors import CORS
from lama_cleaner.helper import (
    download_model,
    load_img,
    numpy_to_bytes,
    pad_img_to_modulo,
)

NUM_THREADS = "4"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS

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
    mask = load_img(input["mask"].read(), gray=True)
    res_np_img = run(image, mask)
    return send_file(
        io.BytesIO(numpy_to_bytes(res_np_img)),
        mimetype="image/png",
        as_attachment=True,
        attachment_filename="result.png",
    )


@app.route("/")
def index():
    return send_file(os.path.join(BUILD_DIR, "index.html"))


def run(image, mask):
    """
    image: [C, H, W]
    """
    origin_height, origin_width = image.shape[1:]
    image = pad_img_to_modulo(image, mod=8)
    mask = pad_img_to_modulo(mask, mod=8)

    mask = (mask > 0) * 1
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    start = time.time()
    inpainted_image = model(image, mask)

    print(
        f"inpainted image shape: {inpainted_image.shape} process time: {(time.time() - start)*1000}ms"
    )
    cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = cur_res[0:origin_height, 0:origin_width, :]
    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    return cur_res


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()


def main():
    global model
    global device
    args = get_args_parser()
    device = torch.device(args.device)
    model_path = download_model()
    model = torch.jit.load(model_path, map_location="cpu")
    model = model.to(device)
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
