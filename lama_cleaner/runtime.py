# https://github.com/huggingface/huggingface_hub/blob/5a12851f54bf614be39614034ed3a9031922d297/src/huggingface_hub/utils/_runtime.py
import platform
import sys
import packaging.version
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
    "xformers",
    "accelerate",
    "lama-cleaner",
    "rembg",
    "realesrgan",
    "gfpgan",
]
# Check once at runtime
for name in _CANDIDATES:
    _package_versions[name] = "N/A"
    try:
        _package_versions[name] = importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        pass


def dump_environment_info() -> Dict[str, str]:
    """Dump information about the machine to help debugging issues. """

    # Generic machine info
    info: Dict[str, Any] = {
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
    }
    info.update(_package_versions)
    print("\n".join([f"- {prop}: {val}" for prop, val in info.items()]) + "\n")
    return info
