import json
from pathlib import Path
from typing import Dict, List


def folder_name_to_show_name(name: str) -> str:
    return name.replace("models--", "").replace("--", "/")


def _scan_models(cache_dir, class_name: List[str]) -> List[str]:
    cache_dir = Path(cache_dir)
    res = []
    for it in cache_dir.glob("**/*/model_index.json"):
        with open(it, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data["_class_name"] in class_name:
                name = folder_name_to_show_name(it.parent.parent.parent.name)
                if name not in res:
                    res.append(name)
    return res


def scan_models(cache_dir) -> Dict[str, List[str]]:
    return {
        "sd": _scan_models(cache_dir, ["StableDiffusionPipeline"]),
        "sd_inpaint": _scan_models(
            cache_dir,
            [
                "StableDiffusionInpaintPipeline",
                "StableDiffusionXLInpaintPipeline",
                "KandinskyV22InpaintPipeline",
            ],
        ),
        "other": _scan_models(
            cache_dir,
            [
                "StableDiffusionInstructPix2PixPipeline",
                "PaintByExamplePipeline",
            ],
        ),
    }
