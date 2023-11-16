import json
from pathlib import Path
from typing import Dict, List


def folder_name_to_show_name(name: str) -> str:
    return name.replace("models--", "").replace("--", "/")


def _scan_models(cache_dir, class_name: str) -> List[str]:
    cache_dir = Path(cache_dir)
    res = []
    for it in cache_dir.glob("**/*/model_index.json"):
        with open(it, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data["_class_name"] == class_name:
                name = folder_name_to_show_name(it.parent.parent.parent.name)
                if name not in res:
                    res.append(name)
    return res


def scan_models(cache_dir) -> List[str]:
    return _scan_models(cache_dir, "StableDiffusionPipeline")


def scan_inpainting_models(cache_dir) -> List[str]:
    return _scan_models(cache_dir, "StableDiffusionInpaintPipeline")
