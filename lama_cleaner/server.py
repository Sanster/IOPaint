#!/usr/bin/env python3

import io
import logging
import multiprocessing
import os
import random
import time
import imghdr
from pathlib import Path
from typing import Union

import cv2
import torch
import numpy as np
from loguru import logger

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
except:
    pass

from flask import Flask, request, send_file, cli, make_response

# Disable ability for Flask to display warning about using a development server in a production environment.
# https://gist.github.com/jerblack/735b9953ba1ab6234abb43174210d356
cli.show_server_banner = lambda *_: None
from flask_cors import CORS

from lama_cleaner.helper import (
    load_img,
    numpy_to_bytes,
    resize_max_size,
)

NUM_THREADS = str(multiprocessing.cpu_count())

# fix libomp problem on windows https://github.com/Sanster/lama-cleaner/issues/56
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
if os.environ.get("CACHE_DIR"):
    os.environ["TORCH_HOME"] = os.environ["CACHE_DIR"]

BUILD_DIR = os.environ.get("LAMA_CLEANER_BUILD_DIR", "app/build")


class NoFlaskwebgui(logging.Filter):
    def filter(self, record):
        return "GET //flaskwebgui-keep-server-alive" not in record.getMessage()


logging.getLogger("werkzeug").addFilter(NoFlaskwebgui())

app = Flask(__name__, static_folder=os.path.join(BUILD_DIR, "static"))
app.config["JSON_AS_ASCII"] = False
CORS(app, expose_headers=["Content-Disposition"])
# MAX_BUFFER_SIZE = 50 * 1000 * 1000  # 50 MB
# async_mode 优先级: eventlet/gevent_uwsgi/gevent/threading
# only threading works on macOS
# socketio = SocketIO(app, max_http_buffer_size=MAX_BUFFER_SIZE, async_mode='threading')

model: ModelManager = None
device = None
input_image_path: str = None


def get_image_ext(img_bytes):
    w = imghdr.what("", img_bytes)
    if w is None:
        w = "jpeg"
    return w


def diffuser_callback(step: int):
    pass
    # socketio.emit('diffusion_step', {'diffusion_step': step})


@app.route("/inpaint", methods=["POST"])
def process():
    input = request.files
    # RGB
    origin_image_bytes = input["image"].read()

    image, alpha_channel = load_img(origin_image_bytes)
    original_shape = image.shape
    interpolation = cv2.INTER_CUBIC

    form = request.form
    size_limit: Union[int, str] = form.get("sizeLimit", "1080")
    if size_limit == "Original":
        size_limit = max(image.shape)
    else:
        size_limit = int(size_limit)

    config = Config(
        ldm_steps=form["ldmSteps"],
        ldm_sampler=form["ldmSampler"],
        hd_strategy=form["hdStrategy"],
        zits_wireframe=form["zitsWireframe"],
        hd_strategy_crop_margin=form["hdStrategyCropMargin"],
        hd_strategy_crop_trigger_size=form["hdStrategyCropTrigerSize"],
        hd_strategy_resize_limit=form["hdStrategyResizeLimit"],
        prompt=form["prompt"],
        use_croper=form["useCroper"],
        croper_x=form["croperX"],
        croper_y=form["croperY"],
        croper_height=form["croperHeight"],
        croper_width=form["croperWidth"],
        sd_mask_blur=form["sdMaskBlur"],
        sd_strength=form["sdStrength"],
        sd_steps=form["sdSteps"],
        sd_guidance_scale=form["sdGuidanceScale"],
        sd_sampler=form["sdSampler"],
        sd_seed=form["sdSeed"],
    )

    if config.sd_seed == -1:
        config.sd_seed = random.randint(1, 9999999)

    logger.info(f"Origin image shape: {original_shape}")
    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
    logger.info(f"Resized image shape: {image.shape}")

    mask, _ = load_img(input["mask"].read(), gray=True)
    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

    start = time.time()
    res_np_img = model(image, mask, config)
    logger.info(f"process time: {(time.time() - start) * 1000}ms")

    torch.cuda.empty_cache()

    if alpha_channel is not None:
        if alpha_channel.shape[:2] != res_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
            )
        res_np_img = np.concatenate(
            (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )

    ext = get_image_ext(origin_image_bytes)

    response = make_response(
        send_file(
            io.BytesIO(numpy_to_bytes(res_np_img, ext)),
            mimetype=f"image/{ext}",
        )
    )
    response.headers["X-Seed"] = str(config.sd_seed)
    return response


@app.route("/model")
def current_model():
    return model.name, 200


@app.route("/model_downloaded/<name>")
def model_downloaded(name):
    return str(model.is_downloaded(name)), 200


@app.route("/model", methods=["POST"])
def switch_model():
    new_name = request.form.get("name")
    if new_name == model.name:
        return "Same model", 200

    try:
        model.switch(new_name)
    except NotImplementedError:
        return f"{new_name} not implemented", 403
    return f"ok, switch to {new_name}", 200


@app.route("/")
def index():
    return send_file(os.path.join(BUILD_DIR, "index.html"))


@app.route("/inputimage")
def set_input_photo():
    if input_image_path:
        with open(input_image_path, "rb") as f:
            image_in_bytes = f.read()
        return send_file(
            input_image_path,
            as_attachment=True,
            attachment_filename=Path(input_image_path).name,
            mimetype=f"image/{get_image_ext(image_in_bytes)}",
        )
    else:
        return "No Input Image"


def main(args):
    global model
    global device
    global input_image_path

    device = torch.device(args.device)
    input_image_path = args.input

    model = ModelManager(
        name=args.model,
        device=device,
        hf_access_token=args.hf_access_token,
        sd_disable_nsfw=args.sd_disable_nsfw,
        sd_cpu_textencoder=args.sd_cpu_textencoder,
        sd_run_local=args.sd_run_local,
        callbacks=[diffuser_callback],
    )

    if args.gui:
        app_width, app_height = args.gui_size
        from flaskwebgui import FlaskUI

        ui = FlaskUI(
            app, width=app_width, height=app_height, host=args.host, port=args.port
        )
        ui.run()
    else:
        # TODO: socketio
        app.run(host=args.host, port=args.port, debug=args.debug)
