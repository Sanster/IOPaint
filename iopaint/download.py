import json
import os
from typing import List

from huggingface_hub.constants import HF_HUB_CACHE
from loguru import logger
from pathlib import Path

from iopaint.const import (
    DEFAULT_MODEL_DIR,
    DIFFUSERS_SD_CLASS_NAME,
    DIFFUSERS_SD_INPAINT_CLASS_NAME,
    DIFFUSERS_SDXL_CLASS_NAME,
    DIFFUSERS_SDXL_INPAINT_CLASS_NAME,
)
from iopaint.model_info import ModelInfo, ModelType


def cli_download_model(model: str):
    from iopaint.model import models
    from iopaint.model.utils import handle_from_pretrained_exceptions

    if model in models and models[model].is_erase_model:
        logger.info(f"Downloading {model}...")
        models[model].download()
        logger.info(f"Done.")
    else:
        logger.info(f"Downloading model from Huggingface: {model}")
        from diffusers import DiffusionPipeline

        downloaded_path = handle_from_pretrained_exceptions(
            DiffusionPipeline.download,
            pretrained_model_name=model,
            variant="fp16",
            resume_download=True,
        )
        logger.info(f"Done. Downloaded to {downloaded_path}")


def folder_name_to_show_name(name: str) -> str:
    return name.replace("models--", "").replace("--", "/")


def scan_single_file_diffusion_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    stable_diffusion_dir = cache_dir / "stable_diffusion"
    stable_diffusion_xl_dir = cache_dir / "stable_diffusion_xl"
    # logger.info(f"Scanning single file sd/sdxl models in {cache_dir}")
    res = []
    for it in stable_diffusion_dir.glob(f"*.*"):
        if it.suffix not in [".safetensors", ".ckpt"]:
            continue
        if "inpaint" in str(it).lower():
            model_type = ModelType.DIFFUSERS_SD_INPAINT
        else:
            model_type = ModelType.DIFFUSERS_SD
        res.append(
            ModelInfo(
                name=it.name,
                path=str(it.absolute()),
                model_type=model_type,
                is_single_file_diffusers=True,
            )
        )

    for it in stable_diffusion_xl_dir.glob(f"*.*"):
        if it.suffix not in [".safetensors", ".ckpt"]:
            continue
        if "inpaint" in str(it).lower():
            model_type = ModelType.DIFFUSERS_SDXL_INPAINT
        else:
            model_type = ModelType.DIFFUSERS_SDXL
        res.append(
            ModelInfo(
                name=it.name,
                path=str(it.absolute()),
                model_type=model_type,
                is_single_file_diffusers=True,
            )
        )
    return res


def scan_inpaint_models(model_dir: Path) -> List[ModelInfo]:
    res = []
    from iopaint.model import models

    # logger.info(f"Scanning inpaint models in {model_dir}")

    for name, m in models.items():
        if m.is_erase_model and m.is_downloaded():
            res.append(
                ModelInfo(
                    name=name,
                    path=name,
                    model_type=ModelType.INPAINT,
                )
            )
    return res


def scan_models() -> List[ModelInfo]:
    model_dir = os.getenv("XDG_CACHE_HOME", DEFAULT_MODEL_DIR)
    available_models = []
    available_models.extend(scan_inpaint_models(model_dir))
    available_models.extend(scan_single_file_diffusion_models(model_dir))
    cache_dir = Path(HF_HUB_CACHE)
    # logger.info(f"Scanning diffusers models in {cache_dir}")
    diffusers_model_names = []
    for it in cache_dir.glob("**/*/model_index.json"):
        with open(it, "r", encoding="utf-8") as f:
            data = json.load(f)
            _class_name = data["_class_name"]
            name = folder_name_to_show_name(it.parent.parent.parent.name)
            if name in diffusers_model_names:
                continue
            if "PowerPaint" in name:
                model_type = ModelType.DIFFUSERS_OTHER
            elif _class_name == DIFFUSERS_SD_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SD
            elif _class_name == DIFFUSERS_SD_INPAINT_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SD_INPAINT
            elif _class_name == DIFFUSERS_SDXL_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SDXL
            elif _class_name == DIFFUSERS_SDXL_INPAINT_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SDXL_INPAINT
            elif _class_name in [
                "StableDiffusionInstructPix2PixPipeline",
                "PaintByExamplePipeline",
                "KandinskyV22InpaintPipeline",
            ]:
                model_type = ModelType.DIFFUSERS_OTHER
            else:
                continue

            diffusers_model_names.append(name)
            available_models.append(
                ModelInfo(
                    name=name,
                    path=name,
                    model_type=model_type,
                )
            )

    return available_models
