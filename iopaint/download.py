import glob
import json
import os
from functools import lru_cache
from typing import List, Optional

from iopaint.schema import ModelType, ModelInfo
from loguru import logger
from pathlib import Path

from iopaint.const import (
    DEFAULT_MODEL_DIR,
    DIFFUSERS_SD_CLASS_NAME,
    DIFFUSERS_SD_INPAINT_CLASS_NAME,
    DIFFUSERS_SDXL_CLASS_NAME,
    DIFFUSERS_SDXL_INPAINT_CLASS_NAME,
    ANYTEXT_NAME,
)
from iopaint.model.original_sd_configs import get_config_files


def cli_download_model(model: str):
    from iopaint.model import models
    from iopaint.model.utils import handle_from_pretrained_exceptions

    if model in models and models[model].is_erase_model:
        logger.info(f"Downloading {model}...")
        models[model].download()
        logger.info("Done.")
    elif model == ANYTEXT_NAME:
        logger.info(f"Downloading {model}...")
        models[model].download()
        logger.info("Done.")
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


@lru_cache(maxsize=512)
def get_sd_model_type(model_abs_path: str) -> Optional[ModelType]:
    if "inpaint" in Path(model_abs_path).name.lower():
        model_type = ModelType.DIFFUSERS_SD_INPAINT
    else:
        # load once to check num_in_channels
        from diffusers import StableDiffusionInpaintPipeline

        try:
            StableDiffusionInpaintPipeline.from_single_file(
                model_abs_path,
                load_safety_checker=False,
                num_in_channels=9,
                original_config_file=get_config_files()['v1']
            )
            model_type = ModelType.DIFFUSERS_SD_INPAINT
        except ValueError as e:
            if "[320, 4, 3, 3]" in str(e):
                model_type = ModelType.DIFFUSERS_SD
            else:
                logger.info(f"Ignore non sdxl file: {model_abs_path}")
                return
        except Exception as e:
            logger.error(f"Failed to load {model_abs_path}: {e}")
            return
    return model_type


@lru_cache()
def get_sdxl_model_type(model_abs_path: str) -> Optional[ModelType]:
    if "inpaint" in model_abs_path:
        model_type = ModelType.DIFFUSERS_SDXL_INPAINT
    else:
        # load once to check num_in_channels
        from diffusers import StableDiffusionXLInpaintPipeline

        try:
            model = StableDiffusionXLInpaintPipeline.from_single_file(
                model_abs_path,
                load_safety_checker=False,
                num_in_channels=9,
                original_config_file=get_config_files()['xl'],
            )
            if model.unet.config.in_channels == 9:
                # https://github.com/huggingface/diffusers/issues/6610
                model_type = ModelType.DIFFUSERS_SDXL_INPAINT
            else:
                model_type = ModelType.DIFFUSERS_SDXL
        except ValueError as e:
            if "[320, 4, 3, 3]" in str(e):
                model_type = ModelType.DIFFUSERS_SDXL
            else:
                logger.info(f"Ignore non sdxl file: {model_abs_path}")
                return
        except Exception as e:
            logger.error(f"Failed to load {model_abs_path}: {e}")
            return
    return model_type


def scan_single_file_diffusion_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    stable_diffusion_dir = cache_dir / "stable_diffusion"
    cache_file = stable_diffusion_dir / "iopaint_cache.json"
    model_type_cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                model_type_cache = json.load(f)
                assert isinstance(model_type_cache, dict)
        except:
            pass

    res = []
    for it in stable_diffusion_dir.glob("*.*"):
        if it.suffix not in [".safetensors", ".ckpt"]:
            continue
        model_abs_path = str(it.absolute())
        model_type = model_type_cache.get(it.name)
        if model_type is None:
            model_type = get_sd_model_type(model_abs_path)
        if model_type is None:
            continue

        model_type_cache[it.name] = model_type
        res.append(
            ModelInfo(
                name=it.name,
                path=model_abs_path,
                model_type=model_type,
                is_single_file_diffusers=True,
            )
        )
    if stable_diffusion_dir.exists():
        with open(cache_file, "w", encoding="utf-8") as fw:
            json.dump(model_type_cache, fw, indent=2, ensure_ascii=False)

    stable_diffusion_xl_dir = cache_dir / "stable_diffusion_xl"
    sdxl_cache_file = stable_diffusion_xl_dir / "iopaint_cache.json"
    sdxl_model_type_cache = {}
    if sdxl_cache_file.exists():
        try:
            with open(sdxl_cache_file, "r", encoding="utf-8") as f:
                sdxl_model_type_cache = json.load(f)
                assert isinstance(sdxl_model_type_cache, dict)
        except:
            pass

    for it in stable_diffusion_xl_dir.glob("*.*"):
        if it.suffix not in [".safetensors", ".ckpt"]:
            continue
        model_abs_path = str(it.absolute())
        model_type = sdxl_model_type_cache.get(it.name)
        if model_type is None:
            model_type = get_sdxl_model_type(model_abs_path)
        if model_type is None:
            continue

        sdxl_model_type_cache[it.name] = model_type
        if stable_diffusion_xl_dir.exists():
            with open(sdxl_cache_file, "w", encoding="utf-8") as fw:
                json.dump(sdxl_model_type_cache, fw, indent=2, ensure_ascii=False)

        res.append(
            ModelInfo(
                name=it.name,
                path=model_abs_path,
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


def scan_diffusers_models() -> List[ModelInfo]:
    from huggingface_hub.constants import HF_HUB_CACHE

    available_models = []
    cache_dir = Path(HF_HUB_CACHE)
    # logger.info(f"Scanning diffusers models in {cache_dir}")
    diffusers_model_names = []
    model_index_files = glob.glob(os.path.join(cache_dir, "**/*", "model_index.json"), recursive=True)
    for it in model_index_files:
        it = Path(it)
        with open(it, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                continue

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
                "AnyText",
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


def _scan_converted_diffusers_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    available_models = []
    diffusers_model_names = []
    model_index_files = glob.glob(os.path.join(cache_dir, "**/*", "model_index.json"), recursive=True)
    for it in model_index_files:
        it = Path(it)
        with open(it, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                logger.error(
                    f"Failed to load {it}, please try revert from original model or fix model_index.json by hand."
                )
                continue

            _class_name = data["_class_name"]
            name = folder_name_to_show_name(it.parent.name)
            if name in diffusers_model_names:
                continue
            elif _class_name == DIFFUSERS_SD_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SD
            elif _class_name == DIFFUSERS_SD_INPAINT_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SD_INPAINT
            elif _class_name == DIFFUSERS_SDXL_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SDXL
            elif _class_name == DIFFUSERS_SDXL_INPAINT_CLASS_NAME:
                model_type = ModelType.DIFFUSERS_SDXL_INPAINT
            else:
                continue

            diffusers_model_names.append(name)
            available_models.append(
                ModelInfo(
                    name=name,
                    path=str(it.parent.absolute()),
                    model_type=model_type,
                )
            )
    return available_models


def scan_converted_diffusers_models(cache_dir) -> List[ModelInfo]:
    cache_dir = Path(cache_dir)
    available_models = []
    stable_diffusion_dir = cache_dir / "stable_diffusion"
    stable_diffusion_xl_dir = cache_dir / "stable_diffusion_xl"
    available_models.extend(_scan_converted_diffusers_models(stable_diffusion_dir))
    available_models.extend(_scan_converted_diffusers_models(stable_diffusion_xl_dir))
    return available_models


def scan_models() -> List[ModelInfo]:
    model_dir = os.getenv("XDG_CACHE_HOME", DEFAULT_MODEL_DIR)
    available_models = []
    available_models.extend(scan_inpaint_models(model_dir))
    available_models.extend(scan_single_file_diffusion_models(model_dir))
    available_models.extend(scan_diffusers_models())
    available_models.extend(scan_converted_diffusers_models(model_dir))
    return available_models
