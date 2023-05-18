import json
import os
from enum import Enum
from pydantic import BaseModel


MPS_SUPPORT_MODELS = [
    "instruct_pix2pix",
    "sd1.5",
    "anything4",
    "realisticVision1.4",
    "sd2",
    "paint_by_example",
    "controlnet",
]

DEFAULT_MODEL = "lama"
AVAILABLE_MODELS = [
    "lama",
    "ldm",
    "zits",
    "mat",
    "fcf",
    "sd1.5",
    "anything4",
    "realisticVision1.4",
    "cv2",
    "manga",
    "sd2",
    "paint_by_example",
    "instruct_pix2pix",
]
SD15_MODELS = ["sd1.5", "anything4", "realisticVision1.4"]

AVAILABLE_DEVICES = ["cuda", "cpu", "mps"]
DEFAULT_DEVICE = "cuda"

NO_HALF_HELP = """
Using full precision model.
If your generate result is always black or green, use this argument. (sd/paint_by_exmaple)
"""

CPU_OFFLOAD_HELP = """
Offloads all models to CPU, significantly reducing vRAM usage. (sd/paint_by_example)
"""

DISABLE_NSFW_HELP = """
Disable NSFW checker. (sd/paint_by_example)
"""

SD_CPU_TEXTENCODER_HELP = """
Run Stable Diffusion text encoder model on CPU to save GPU memory.
"""

SD_CONTROLNET_HELP = """
Run Stable Diffusion inpainting model with ControlNet. You can switch control method in webui.
"""
DEFAULT_CONTROLNET_METHOD = "control_v11p_sd15_canny"
SD_CONTROLNET_CHOICES = [
    "control_v11p_sd15_canny",
    "control_v11p_sd15_openpose",
    "control_v11p_sd15_inpaint",
    "control_v11f1p_sd15_depth"
]

SD_LOCAL_MODEL_HELP = """
Load Stable Diffusion 1.5 model(ckpt/safetensors) from local path.
"""

LOCAL_FILES_ONLY_HELP = """
Use local files only, not connect to Hugging Face server. (sd/paint_by_example)
"""

ENABLE_XFORMERS_HELP = """
Enable xFormers optimizations. Requires xformers package has been installed. See: https://github.com/facebookresearch/xformers (sd/paint_by_example)
"""

DEFAULT_MODEL_DIR = os.getenv(
    "XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")
)
MODEL_DIR_HELP = """
Model download directory (by setting XDG_CACHE_HOME environment variable), by default model downloaded to ~/.cache
"""

OUTPUT_DIR_HELP = """
Result images will be saved to output directory automatically without confirmation.
"""

INPUT_HELP = """
If input is image, it will be loaded by default.
If input is directory, you can browse and select image in file manager.
"""

GUI_HELP = """
Launch Lama Cleaner as desktop app
"""

NO_GUI_AUTO_CLOSE_HELP = """
Prevent backend auto close after the GUI window closed.
"""

QUALITY_HELP = """
Quality of image encoding, 0-100. Default is 95, higher quality will generate larger file size.
"""


class RealESRGANModelName(str, Enum):
    realesr_general_x4v3 = "realesr-general-x4v3"
    RealESRGAN_x4plus = "RealESRGAN_x4plus"
    RealESRGAN_x4plus_anime_6B = "RealESRGAN_x4plus_anime_6B"


RealESRGANModelNameList = [e.value for e in RealESRGANModelName]

INTERACTIVE_SEG_HELP = "Enable interactive segmentation using Segment Anything."
INTERACTIVE_SEG_MODEL_HELP = "Model size: vit_b < vit_l < vit_h. Bigger model size means better segmentation but slower speed."
AVAILABLE_INTERACTIVE_SEG_MODELS = ["vit_b", "vit_l", "vit_h"]
AVAILABLE_INTERACTIVE_SEG_DEVICES = ["cuda", "cpu", "mps"]
REMOVE_BG_HELP = "Enable remove background. Always run on CPU"
ANIMESEG_HELP = "Enable anime segmentation. Always run on CPU"
REALESRGAN_HELP = "Enable realesrgan super resolution"
REALESRGAN_AVAILABLE_DEVICES = ["cpu", "cuda", "mps"]
GFPGAN_HELP = (
    "Enable GFPGAN face restore. To enhance background, use with --enable-realesrgan"
)
GFPGAN_AVAILABLE_DEVICES = ["cpu", "cuda", "mps"]
RESTOREFORMER_HELP = "Enable RestoreFormer face restore. To enhance background, use with --enable-realesrgan"
RESTOREFORMER_AVAILABLE_DEVICES = ["cpu", "cuda", "mps"]
GIF_HELP = "Enable GIF plugin. Make GIF to compare original and cleaned image"


class Config(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    model: str = DEFAULT_MODEL
    sd_local_model_path: str = None
    sd_controlnet: bool = False
    sd_controlnet_method: str = DEFAULT_CONTROLNET_METHOD
    device: str = DEFAULT_DEVICE
    gui: bool = False
    no_gui_auto_close: bool = False
    no_half: bool = False
    cpu_offload: bool = False
    disable_nsfw: bool = False
    sd_cpu_textencoder: bool = False
    enable_xformers: bool = False
    local_files_only: bool = False
    model_dir: str = DEFAULT_MODEL_DIR
    input: str = None
    output_dir: str = None
    # plugins
    enable_interactive_seg: bool = False
    interactive_seg_model: str = "vit_l"
    interactive_seg_device: str = "cpu"
    enable_remove_bg: bool = False
    enable_anime_seg: bool = False
    enable_realesrgan: bool = False
    realesrgan_device: str = "cpu"
    realesrgan_model: str = RealESRGANModelName.realesr_general_x4v3.value
    realesrgan_no_half: bool = False
    enable_gfpgan: bool = False
    gfpgan_device: str = "cpu"
    enable_restoreformer: bool = False
    restoreformer_device: str = "cpu"
    enable_gif: bool = False


def load_config(installer_config: str):
    if os.path.exists(installer_config):
        with open(installer_config, "r", encoding="utf-8") as f:
            return Config(**json.load(f))
    else:
        return Config()
