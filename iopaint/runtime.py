# https://github.com/huggingface/huggingface_hub/blob/5a12851f54bf614be39614034ed3a9031922d297/src/huggingface_hub/utils/_runtime.py
import os
import platform
import sys
from pathlib import Path

import packaging.version
from iopaint.schema import Device
from loguru import logger
from rich import print
from typing import Dict, Any


_PY_VERSION: str = sys.version.split()[0].rstrip("+")

if packaging.version.Version(_PY_VERSION) < packaging.version.Version("3.8.0"):
    import importlib_metadata  # type: ignore
else:
    import importlib.metadata as importlib_metadata  # type: ignore

_package_versions = {}

_CANDIDATES = [
    "torch",
    "torchvision",
    "Pillow",
    "diffusers",
    "transformers",
    "opencv-python",
    "accelerate",
    "iopaint",
    "rembg",
    "onnxruntime",
]
# Check once at runtime
for name in _CANDIDATES:
    _package_versions[name] = "N/A"
    try:
        _package_versions[name] = importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        pass


def dump_environment_info() -> Dict[str, str]:
    """Dump information about the machine to help debugging issues."""

    # Generic machine info
    info: Dict[str, Any] = {
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
    }
    info.update(_package_versions)
    print("\n".join([f"- {prop}: {val}" for prop, val in info.items()]) + "\n")
    return info


def check_device(device: Device) -> Device:
    if device == Device.cuda:
        import platform

        if platform.system() == "Darwin":
            logger.warning("MacOS does not support cuda, use cpu instead")
            return Device.cpu
        else:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA is not available, use cpu instead")
                return Device.cpu
    elif device == Device.mps:
        import torch

        if not torch.backends.mps.is_available():
            logger.warning("mps is not available, use cpu instead")
            return Device.cpu
    return device


def setup_model_dir(model_dir: Path):
    model_dir = model_dir.expanduser().absolute()
    logger.info(f"Model directory: {model_dir}")
    os.environ["U2NET_HOME"] = str(model_dir)
    os.environ["XDG_CACHE_HOME"] = str(model_dir)
    if not model_dir.exists():
        logger.info(f"Create model directory: {model_dir}")
        model_dir.mkdir(exist_ok=True, parents=True)
    return model_dir
