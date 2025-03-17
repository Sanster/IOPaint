import asyncio
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, List

import cv2
import numpy as np
import socketio
import torch

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
    torch._C._jit_set_profiling_mode(False)
except:
    pass

import uvicorn
from PIL import Image
from fastapi import APIRouter, FastAPI, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from socketio import AsyncServer

from iopaint.file_manager import FileManager
from iopaint.helper import (
    load_img,
    decode_base64_to_image,
    pil_to_bytes,
    numpy_to_bytes,
    concat_alpha_channel,
    gen_frontend_mask,
    adjust_mask,
)
from iopaint.model.utils import torch_gc
from iopaint.model_manager import ModelManager
from iopaint.plugins import build_plugins, RealESRGANUpscaler, InteractiveSeg
from iopaint.plugins.base_plugin import BasePlugin
from iopaint.plugins.remove_bg import RemoveBG
from iopaint.schema import (
    GenInfoResponse,
    ApiConfig,
    ServerConfigResponse,
    SwitchModelRequest,
    InpaintRequest,
    RunPluginRequest,
    SDSampler,
    PluginInfo,
    AdjustMaskRequest,
    RemoveBGModel,
    SwitchPluginModelRequest,
    ModelInfo,
    InteractiveSegModel,
    RealESRGANModel,
)

CURRENT_DIR = Path(__file__).parent.absolute().resolve()
WEB_APP_DIR = CURRENT_DIR / "web_app"


def api_middleware(app: FastAPI):
    rich_available = False
    try:
        if os.environ.get("WEBUI_RICH_EXCEPTIONS", None) is not None:
            import anyio  # importing just so it can be placed on silent list
            import starlette  # importing just so it can be placed on silent list
            from rich.console import Console

            console = Console()
            rich_available = True
    except Exception:
        pass

    def handle_exception(request: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get("detail", ""),
            "body": vars(e).get("body", ""),
            "errors": str(e),
        }
        if not isinstance(
            e, HTTPException
        ):  # do not print backtrace on known httpexceptions
            message = f"API error: {request.method}: {request.url} {err}"
            if rich_available:
                print(message)
                console.print_exception(
                    show_locals=True,
                    max_frames=2,
                    extra_lines=1,
                    suppress=[anyio, starlette],
                    word_wrap=False,
                    width=min([console.width, 200]),
                )
            else:
                traceback.print_exc()
        return JSONResponse(
            status_code=vars(e).get("status_code", 500), content=jsonable_encoder(err)
        )

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)

    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_origins": ["*"],
        "allow_credentials": True,
        "expose_headers": ["X-Seed"],
    }
    app.add_middleware(CORSMiddleware, **cors_options)


global_sio: AsyncServer = None


def diffuser_callback(pipe, step: int, timestep: int, callback_kwargs: Dict = {}):
    # self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict
    # logger.info(f"diffusion callback: step={step}, timestep={timestep}")

    # We use asyncio loos for task processing. Perhaps in the future, we can add a processing queue similar to InvokeAI,
    # but for now let's just start a separate event loop. It shouldn't make a difference for single person use
    asyncio.run(global_sio.emit("diffusion_progress", {"step": step}))
    return {}


class Api:
    def __init__(self, app: FastAPI, config: ApiConfig):
        self.app = app
        self.config = config
        self.router = APIRouter()
        self.queue_lock = threading.Lock()
        api_middleware(self.app)

        self.file_manager = self._build_file_manager()
        self.plugins = self._build_plugins()
        self.model_manager = self._build_model_manager()

        # fmt: off
        self.add_api_route("/api/v1/gen-info", self.api_geninfo, methods=["POST"], response_model=GenInfoResponse)
        self.add_api_route("/api/v1/server-config", self.api_server_config, methods=["GET"],
                           response_model=ServerConfigResponse)
        self.add_api_route("/api/v1/model", self.api_current_model, methods=["GET"], response_model=ModelInfo)
        self.add_api_route("/api/v1/model", self.api_switch_model, methods=["POST"], response_model=ModelInfo)
        self.add_api_route("/api/v1/inputimage", self.api_input_image, methods=["GET"])
        self.add_api_route("/api/v1/inpaint", self.api_inpaint, methods=["POST"])
        self.add_api_route("/api/v1/switch_plugin_model", self.api_switch_plugin_model, methods=["POST"])
        self.add_api_route("/api/v1/run_plugin_gen_mask", self.api_run_plugin_gen_mask, methods=["POST"])
        self.add_api_route("/api/v1/run_plugin_gen_image", self.api_run_plugin_gen_image, methods=["POST"])
        self.add_api_route("/api/v1/samplers", self.api_samplers, methods=["GET"])
        self.add_api_route("/api/v1/adjust_mask", self.api_adjust_mask, methods=["POST"])
        self.add_api_route("/api/v1/save_image", self.api_save_image, methods=["POST"])
        self.app.mount("/", StaticFiles(directory=WEB_APP_DIR, html=True), name="assets")
        # fmt: on

        global global_sio
        self.sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self.combined_asgi_app = socketio.ASGIApp(self.sio, self.app)
        self.app.mount("/ws", self.combined_asgi_app)
        global_sio = self.sio

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    def api_save_image(self, file: UploadFile):
        # Sanitize filename to prevent path traversal
        safe_filename = Path(file.filename).name  # Get just the filename component

        # Construct the full path within output_dir
        output_path = self.config.output_dir / safe_filename

        # Ensure output directory exists
        if not self.config.output_dir or not self.config.output_dir.exists():
            raise HTTPException(
                status_code=400,
                detail="Output directory not configured or doesn't exist",
            )

        # Read and write the file
        origin_image_bytes = file.file.read()
        with open(output_path, "wb") as fw:
            fw.write(origin_image_bytes)

    def api_current_model(self) -> ModelInfo:
        return self.model_manager.current_model

    def api_switch_model(self, req: SwitchModelRequest) -> ModelInfo:
        if req.name == self.model_manager.name:
            return self.model_manager.current_model
        self.model_manager.switch(req.name)
        return self.model_manager.current_model

    def api_switch_plugin_model(self, req: SwitchPluginModelRequest):
        if req.plugin_name in self.plugins:
            self.plugins[req.plugin_name].switch_model(req.model_name)
            if req.plugin_name == RemoveBG.name:
                self.config.remove_bg_model = req.model_name
            if req.plugin_name == RealESRGANUpscaler.name:
                self.config.realesrgan_model = req.model_name
            if req.plugin_name == InteractiveSeg.name:
                self.config.interactive_seg_model = req.model_name
            torch_gc()

    def api_server_config(self) -> ServerConfigResponse:
        plugins = []
        for it in self.plugins.values():
            plugins.append(
                PluginInfo(
                    name=it.name,
                    support_gen_image=it.support_gen_image,
                    support_gen_mask=it.support_gen_mask,
                )
            )

        return ServerConfigResponse(
            plugins=plugins,
            modelInfos=self.model_manager.scan_models(),
            removeBGModel=self.config.remove_bg_model,
            removeBGModels=RemoveBGModel.values(),
            realesrganModel=self.config.realesrgan_model,
            realesrganModels=RealESRGANModel.values(),
            interactiveSegModel=self.config.interactive_seg_model,
            interactiveSegModels=InteractiveSegModel.values(),
            enableFileManager=self.file_manager is not None,
            enableAutoSaving=self.config.output_dir is not None,
            enableControlnet=self.model_manager.enable_controlnet,
            controlnetMethod=self.model_manager.controlnet_method,
            disableModelSwitch=False,
            isDesktop=False,
            samplers=self.api_samplers(),
        )

    def api_input_image(self) -> FileResponse:
        if self.config.input is None:
            raise HTTPException(status_code=200, detail="No input image configured")

        if self.config.input.is_file():
            return FileResponse(self.config.input)
        raise HTTPException(status_code=404, detail="Input image not found")

    def api_geninfo(self, file: UploadFile) -> GenInfoResponse:
        _, _, info = load_img(file.file.read(), return_info=True)
        parts = info.get("parameters", "").split("Negative prompt: ")
        prompt = parts[0].strip()
        negative_prompt = ""
        if len(parts) > 1:
            negative_prompt = parts[1].split("\n")[0].strip()
        return GenInfoResponse(prompt=prompt, negative_prompt=negative_prompt)

    def api_inpaint(self, req: InpaintRequest):
        image, alpha_channel, infos, ext = decode_base64_to_image(req.image)
        mask, _, _, _ = decode_base64_to_image(req.mask, gray=True)
        logger.info(f"image ext: {ext}")

        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        if image.shape[:2] != mask.shape[:2]:
            raise HTTPException(
                400,
                detail=f"Image size({image.shape[:2]}) and mask size({mask.shape[:2]}) not match.",
            )

        start = time.time()
        rgb_np_img = self.model_manager(image, mask, req)
        logger.info(f"process time: {(time.time() - start) * 1000:.2f}ms")
        torch_gc()

        rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)

        res_img_bytes = pil_to_bytes(
            Image.fromarray(rgb_res),
            ext=ext,
            quality=self.config.quality,
            infos=infos,
        )

        asyncio.run(self.sio.emit("diffusion_finish"))

        return Response(
            content=res_img_bytes,
            media_type=f"image/{ext}",
            headers={"X-Seed": str(req.sd_seed)},
        )

    def api_run_plugin_gen_image(self, req: RunPluginRequest):
        ext = "png"
        if req.name not in self.plugins:
            raise HTTPException(status_code=422, detail="Plugin not found")
        if not self.plugins[req.name].support_gen_image:
            raise HTTPException(
                status_code=422, detail="Plugin does not support output image"
            )
        rgb_np_img, alpha_channel, infos, _ = decode_base64_to_image(req.image)
        bgr_or_rgba_np_img = self.plugins[req.name].gen_image(rgb_np_img, req)
        torch_gc()

        if bgr_or_rgba_np_img.shape[2] == 4:
            rgba_np_img = bgr_or_rgba_np_img
        else:
            rgba_np_img = cv2.cvtColor(bgr_or_rgba_np_img, cv2.COLOR_BGR2RGB)
            rgba_np_img = concat_alpha_channel(rgba_np_img, alpha_channel)

        return Response(
            content=pil_to_bytes(
                Image.fromarray(rgba_np_img),
                ext=ext,
                quality=self.config.quality,
                infos=infos,
            ),
            media_type=f"image/{ext}",
        )

    def api_run_plugin_gen_mask(self, req: RunPluginRequest):
        if req.name not in self.plugins:
            raise HTTPException(status_code=422, detail="Plugin not found")
        if not self.plugins[req.name].support_gen_mask:
            raise HTTPException(
                status_code=422, detail="Plugin does not support output image"
            )
        rgb_np_img, _, _, _ = decode_base64_to_image(req.image)
        bgr_or_gray_mask = self.plugins[req.name].gen_mask(rgb_np_img, req)
        torch_gc()
        res_mask = gen_frontend_mask(bgr_or_gray_mask)
        return Response(
            content=numpy_to_bytes(res_mask, "png"),
            media_type="image/png",
        )

    def api_samplers(self) -> List[str]:
        return [member.value for member in SDSampler.__members__.values()]

    def api_adjust_mask(self, req: AdjustMaskRequest):
        mask, _, _, _ = decode_base64_to_image(req.mask, gray=True)
        mask = adjust_mask(mask, req.kernel_size, req.operate)
        return Response(content=numpy_to_bytes(mask, "png"), media_type="image/png")

    def launch(self):
        self.app.include_router(self.router)
        uvicorn.run(
            self.combined_asgi_app,
            host=self.config.host,
            port=self.config.port,
            timeout_keep_alive=999999999,
        )

    def _build_file_manager(self) -> Optional[FileManager]:
        if self.config.input and self.config.input.is_dir():
            logger.info(
                f"Input is directory, initialize file manager {self.config.input}"
            )

            return FileManager(
                app=self.app,
                input_dir=self.config.input,
                mask_dir=self.config.mask_dir,
                output_dir=self.config.output_dir,
            )
        return None

    def _build_plugins(self) -> Dict[str, BasePlugin]:
        return build_plugins(
            self.config.enable_interactive_seg,
            self.config.interactive_seg_model,
            self.config.interactive_seg_device,
            self.config.enable_remove_bg,
            self.config.remove_bg_device,
            self.config.remove_bg_model,
            self.config.enable_anime_seg,
            self.config.enable_realesrgan,
            self.config.realesrgan_device,
            self.config.realesrgan_model,
            self.config.enable_gfpgan,
            self.config.gfpgan_device,
            self.config.enable_restoreformer,
            self.config.restoreformer_device,
            self.config.no_half,
        )

    def _build_model_manager(self):
        return ModelManager(
            name=self.config.model,
            device=torch.device(self.config.device),
            no_half=self.config.no_half,
            low_mem=self.config.low_mem,
            disable_nsfw=self.config.disable_nsfw_checker,
            sd_cpu_textencoder=self.config.cpu_textencoder,
            local_files_only=self.config.local_files_only,
            cpu_offload=self.config.cpu_offload,
            callback=diffuser_callback,
        )
