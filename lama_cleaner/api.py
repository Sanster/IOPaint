import os
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, List

import cv2
import torch
import numpy as np
from loguru import logger
from PIL import Image

import uvicorn
from fastapi import APIRouter, FastAPI, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from lama_cleaner.helper import (
    load_img,
    decode_base64_to_image,
    pil_to_bytes,
    numpy_to_bytes,
    concat_alpha_channel,
)
from lama_cleaner.model.utils import torch_gc
from lama_cleaner.model_info import ModelInfo
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.plugins import build_plugins, InteractiveSeg, RemoveBG, AnimeSeg
from lama_cleaner.schema import (
    GenInfoResponse,
    ApiConfig,
    ServerConfigResponse,
    SwitchModelRequest,
    InpaintRequest,
    RunPluginRequest,
)
from lama_cleaner.file_manager import FileManager

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
    }
    app.add_middleware(CORSMiddleware, **cors_options)


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
        self.add_api_route("/api/v1/server-config", self.api_server_config, methods=["GET"], response_model=ServerConfigResponse)
        self.add_api_route("/api/v1/models", self.api_models, methods=["GET"], response_model=List[ModelInfo])
        self.add_api_route("/api/v1/model", self.api_current_model, methods=["GET"], response_model=ModelInfo)
        self.add_api_route("/api/v1/model", self.api_switch_model, methods=["POST"], response_model=ModelInfo)
        self.add_api_route("/api/v1/inputimage", self.api_input_image, methods=["GET"])
        self.add_api_route("/api/v1/inpaint", self.api_inpaint, methods=["POST"])
        self.add_api_route("/api/v1/run_plugin", self.api_run_plugin, methods=["POST"])
        self.app.mount("/", StaticFiles(directory=WEB_APP_DIR, html=True), name="assets")
        # fmt: on

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    def api_models(self) -> List[ModelInfo]:
        return self.model_manager.scan_models()

    def api_current_model(self) -> ModelInfo:
        return self.model_manager.current_model

    def api_switch_model(self, req: SwitchModelRequest) -> ModelInfo:
        if req.name == self.model_manager.name:
            return self.model_manager.current_model
        self.model_manager.switch(req.name)
        return self.model_manager.current_model

    def api_server_config(self) -> ServerConfigResponse:
        return ServerConfigResponse(
            plugins=list(self.plugins.keys()),
            enableFileManager=self.file_manager is not None,
            enableAutoSaving=self.config.output_dir is not None,
            enableControlnet=self.model_manager.enable_controlnet,
            controlnetMethod=self.model_manager.controlnet_method,
            disableModelSwitch=self.config.disable_model_switch,
            isDesktop=self.config.gui,
        )

    def api_input_image(self) -> FileResponse:
        if self.config.input and self.config.input.is_file():
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
        image, alpha_channel, infos = decode_base64_to_image(req.image)
        mask, _, _ = decode_base64_to_image(req.mask, gray=True)

        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        if image.shape[:2] != mask.shape[:2]:
            raise HTTPException(
                400,
                detail=f"Image size({image.shape[:2]}) and mask size({mask.shape[:2]}) not match.",
            )

        if req.paint_by_example_example_image:
            paint_by_example_image, _, _ = decode_base64_to_image(
                req.paint_by_example_example_image
            )

        start = time.time()
        rgb_np_img = self.model_manager(image, mask, req)
        logger.info(f"process time: {(time.time() - start) * 1000:.2f}ms")
        torch_gc()

        rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)

        ext = "png"
        res_img_bytes = pil_to_bytes(
            Image.fromarray(rgb_res),
            ext=ext,
            quality=self.config.quality,
            infos=infos,
        )
        return Response(
            content=res_img_bytes,
            media_type=f"image/{ext}",
            headers={"X-Seed": str(req.sd_seed)},
        )

    def api_run_plugin(self, req: RunPluginRequest):
        if req.name not in self.plugins:
            raise HTTPException(status_code=404, detail="Plugin not found")
        image, alpha_channel, infos = decode_base64_to_image(req.image)
        bgr_res = self.plugins[req.name].run(image, req)
        torch_gc()
        if req.name == InteractiveSeg.name:
            return Response(
                content=numpy_to_bytes(bgr_res, "png"),
                media_type="image/png",
            )
        ext = "png"
        if req.name in [RemoveBG.name, AnimeSeg.name]:
            rgb_res = bgr_res
        else:
            rgb_res = cv2.cvtColor(bgr_res, cv2.COLOR_BGR2RGB)
            rgb_res = concat_alpha_channel(rgb_res, alpha_channel)

        return Response(
            content=pil_to_bytes(
                Image.fromarray(rgb_res),
                ext=ext,
                quality=self.config.quality,
                infos=infos,
            ),
            media_type=f"image/{ext}",
        )

    def launch(self):
        self.app.include_router(self.router)
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            timeout_keep_alive=60,
        )

    def _build_file_manager(self) -> Optional[FileManager]:
        if self.config.input and self.config.input.is_dir():
            logger.info(
                f"Input is directory, initialize file manager {self.config.input}"
            )

            return FileManager(
                app=self.app,
                input_dir=self.config.input,
                output_dir=self.config.output_dir,
            )
        return None

    def _build_plugins(self) -> Dict:
        return build_plugins(
            self.config.enable_interactive_seg,
            self.config.interactive_seg_model,
            self.config.interactive_seg_device,
            self.config.enable_remove_bg,
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
            disable_nsfw=self.config.disable_nsfw_checker,
            sd_cpu_textencoder=self.config.cpu_textencoder,
            cpu_offload=self.config.cpu_offload,
        )


if __name__ == "__main__":
    from lama_cleaner.schema import InteractiveSegModel, RealESRGANModel

    app = FastAPI()
    api = Api(
        app,
        ApiConfig(
            host="127.0.0.1",
            port=8080,
            model="lama",
            no_half=False,
            cpu_offload=False,
            disable_nsfw_checker=False,
            cpu_textencoder=False,
            device="cpu",
            gui=False,
            disable_model_switch=False,
            input="/Users/cwq/code/github/MI-GAN/examples/places2_512_object/images",
            output_dir="/Users/cwq/code/github/lama-cleaner/tmp",
            quality=100,
            enable_interactive_seg=False,
            interactive_seg_model=InteractiveSegModel.vit_b,
            interactive_seg_device="cpu",
            enable_remove_bg=False,
            enable_anime_seg=False,
            enable_realesrgan=False,
            realesrgan_device="cpu",
            realesrgan_model=RealESRGANModel.realesr_general_x4v3,
            enable_gfpgan=False,
            gfpgan_device="cpu",
            enable_restoreformer=False,
            restoreformer_device="cpu",
        ),
    )
    api.launch()