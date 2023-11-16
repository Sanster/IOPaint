import os

from loguru import logger
from pathlib import Path


def cli_download_model(model: str, model_dir: str):
    if os.path.isfile(model_dir):
        raise ValueError(f"invalid --model-dir: {model_dir} is a file")

    if not os.path.exists(model_dir):
        logger.info(f"Create model cache directory: {model_dir}")
        Path(model_dir).mkdir(exist_ok=True, parents=True)

    os.environ["XDG_CACHE_HOME"] = model_dir

    from lama_cleaner.model_manager import models

    if model in models:
        logger.info(f"Downloading {model}...")
        models[model].download()
        logger.info(f"Done.")
    else:
        logger.info(f"Downloading model from Huggingface: {model}")
