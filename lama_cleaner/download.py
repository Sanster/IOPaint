import json
import os
from typing import List

from loguru import logger
from pathlib import Path

from lama_cleaner.const import DIFFUSERS_MODEL_FP16_REVERSION, DEFAULT_MODEL_DIR
from lama_cleaner.runtime import setup_model_dir
from lama_cleaner.schema import (
    ModelInfo,
    ModelType,
    DIFFUSERS_SD_INPAINT_CLASS_NAME,
    DIFFUSERS_SDXL_INPAINT_CLASS_NAME,
    DIFFUSERS_SD_CLASS_NAME,
    DIFFUSERS_SDXL_CLASS_NAME,
)


def cli_download_model(model: str, model_dir: Path):
    setup_model_dir(model_dir)
    from lama_cleaner.model import models

    if model in models:
        logger.info(f"Downloading {model}...")
        models[model].download()
        logger.info(f"Done.")
    else:
        logger.info(f"Downloading model from Huggingface: {model}")
        from diffusers import DiffusionPipeline

        downloaded_path = DiffusionPipeline.download(
            pretrained_model_name=model,
            variant="fp16" if model in DIFFUSERS_MODEL_FP16_REVERSION else "main",
            resume_download=True,
        )
        logger.info(f"Done. Downloaded to {downloaded_path}")


def folder_name_to_show_name(name: str) -> str:
    return name.replace("models--", "").replace("--", "/")


def scan_single_file_diffusion_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    res = []
    for it in cache_dir.glob(f"*.*"):
        if it.suffix not in [".safetensors", ".ckpt"]:
            continue
        if "inpaint" in str(it).lower():
            if "sdxl" in str(it).lower():
                model_type = ModelType.DIFFUSERS_SDXL_INPAINT
            else:
                model_type = ModelType.DIFFUSERS_SD_INPAINT
        else:
            if "sdxl" in str(it).lower():
                model_type = ModelType.DIFFUSERS_SDXL
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
    return res


def scan_inpaint_models() -> List[ModelInfo]:
    res = []
    from lama_cleaner.model import models

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
    from diffusers.utils import DIFFUSERS_CACHE

    available_models = []
    available_models.extend(scan_inpaint_models())
    available_models.extend(scan_single_file_diffusion_models(DEFAULT_MODEL_DIR))
    cache_dir = Path(DIFFUSERS_CACHE)
    diffusers_model_names = []
    for it in cache_dir.glob("**/*/model_index.json"):
        with open(it, "r", encoding="utf-8") as f:
            data = json.load(f)
            _class_name = data["_class_name"]
            name = folder_name_to_show_name(it.parent.parent.parent.name)
            if name in diffusers_model_names:
                continue

            if _class_name == DIFFUSERS_SD_CLASS_NAME:
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
