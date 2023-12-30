#!/usr/bin/env python3
import multiprocessing
import os

import cv2

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

NUM_THREADS = str(multiprocessing.cpu_count())
cv2.setNumThreads(NUM_THREADS)

# fix libomp problem on windows https://github.com/Sanster/lama-cleaner/issues/56
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS

import hashlib
import traceback
from dataclasses import dataclass

import io
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from lama_cleaner.const import *
from lama_cleaner.file_manager import FileManager
from lama_cleaner.model.utils import torch_gc
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.plugins import (
    InteractiveSeg,
    RemoveBG,
    AnimeSeg,
    build_plugins,
)
from lama_cleaner.schema import InpaintRequest
from lama_cleaner.helper import (
    load_img,
    numpy_to_bytes,
    resize_max_size,
    pil_to_bytes,
    is_mac,
    get_image_ext, concat_alpha_channel,
)

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
except:
    pass


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BUILD_DIR = os.environ.get("LAMA_CLEANER_BUILD_DIR", "app/build")

global_config = GlobalConfig()

def diffuser_callback(i, t, latents):
    socketio.emit("diffusion_progress", {"step": i})


def start(
    host: str,
    port: int,
    model: str,
    no_half: bool,
    cpu_offload: bool,
    disable_nsfw_checker,
    cpu_textencoder: bool,
    device: Device,
    gui: bool,
    disable_model_switch: bool,
    input: Path,
    output_dir: Path,
    quality: int,
    enable_interactive_seg: bool,
    interactive_seg_model: InteractiveSegModel,
    interactive_seg_device: Device,
    enable_remove_bg: bool,
    enable_anime_seg: bool,
    enable_realesrgan: bool,
    realesrgan_device: Device,
    realesrgan_model: RealESRGANModel,
    enable_gfpgan: bool,
    gfpgan_device: Device,
    enable_restoreformer: bool,
    restoreformer_device: Device,
):
    if input:
        if not input.exists():
            logger.error(f"invalid --input: {input} not exists")
            exit()
        if input.is_dir():
            logger.info(f"Initialize file manager")
            file_manager = FileManager(app)
            app.config["THUMBNAIL_MEDIA_ROOT"] = input
            app.config["THUMBNAIL_MEDIA_THUMBNAIL_ROOT"] = os.path.join(
                output_dir, "lama_cleaner_thumbnails"
            )
            file_manager.output_dir = output_dir
            global_config.file_manager = file_manager
        else:
            global_config.input_image_path = input

    global_config.image_quality = quality
    global_config.disable_model_switch = disable_model_switch
    global_config.is_desktop = gui
    build_plugins(
        global_config,
        enable_interactive_seg,
        interactive_seg_model,
        interactive_seg_device,
        enable_remove_bg,
        enable_anime_seg,
        enable_realesrgan,
        realesrgan_device,
        realesrgan_model,
        enable_gfpgan,
        gfpgan_device,
        enable_restoreformer,
        restoreformer_device,
        no_half,
    )
    if output_dir:
        output_dir = output_dir.expanduser().absolute()
        logger.info(f"Image will auto save to output dir: {output_dir}")
        if not output_dir.exists():
            logger.info(f"Create output dir: {output_dir}")
            output_dir.mkdir(parents=True)
        global_config.output_dir = output_dir

    global_config.model_manager = ModelManager(
        name=model,
        device=torch.device(device),
        no_half=no_half,
        disable_nsfw=disable_nsfw_checker,
        sd_cpu_textencoder=cpu_textencoder,
        cpu_offload=cpu_offload,
        callback=diffuser_callback,
    )

    if gui:
        from flaskwebgui import FlaskUI

        ui = FlaskUI(
            app,
            socketio=socketio,
            width=1200,
            height=800,
            host=host,
            port=port,
            close_server_on_exit=True,
            idle_interval=60,
        )
        ui.run()
    else:
        socketio.run(
            app,
            host=host,
            port=port,
            allow_unsafe_werkzeug=True,
        )
