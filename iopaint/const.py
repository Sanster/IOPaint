import json
import os
from enum import Enum
from pydantic import BaseModel

INSTRUCT_PIX2PIX_NAME = "timbrooks/instruct-pix2pix"
KANDINSKY22_NAME = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
POWERPAINT_NAME = "Sanster/PowerPaint-V1-stable-diffusion-inpainting"


DIFFUSERS_SD_CLASS_NAME = "StableDiffusionPipeline"
DIFFUSERS_SD_INPAINT_CLASS_NAME = "StableDiffusionInpaintPipeline"
DIFFUSERS_SDXL_CLASS_NAME = "StableDiffusionXLPipeline"
DIFFUSERS_SDXL_INPAINT_CLASS_NAME = "StableDiffusionXLInpaintPipeline"

MPS_UNSUPPORT_MODELS = [
    "lama",
    "ldm",
    "zits",
    "mat",
    "fcf",
    "cv2",
    "manga",
]

DEFAULT_MODEL = "lama"
AVAILABLE_MODELS = ["lama", "ldm", "zits", "mat", "fcf", "manga", "cv2", "migan"]


AVAILABLE_DEVICES = ["cuda", "cpu", "mps"]
DEFAULT_DEVICE = "cuda"

NO_HALF_HELP = """
Using full precision(fp32) model.
If your diffusion model generate result is always black or green, use this argument.
"""

CPU_OFFLOAD_HELP = """
Offloads diffusion model's weight to CPU RAM, significantly reducing vRAM usage.
"""

DISABLE_NSFW_HELP = """
Disable NSFW checker for diffusion model.
"""

CPU_TEXTENCODER_HELP = """
Run diffusion models text encoder on CPU to reduce vRAM usage.
"""

SD_CONTROLNET_CHOICES = [
    "lllyasviel/control_v11p_sd15_canny",
    # "lllyasviel/control_v11p_sd15_seg",
    "lllyasviel/control_v11p_sd15_openpose",
    "lllyasviel/control_v11p_sd15_inpaint",
    "lllyasviel/control_v11f1p_sd15_depth",
]

SD2_CONTROLNET_CHOICES = [
    "thibaud/controlnet-sd21-canny-diffusers",
    "thibaud/controlnet-sd21-depth-diffusers",
    "thibaud/controlnet-sd21-openpose-diffusers",
]

SDXL_CONTROLNET_CHOICES = [
    "thibaud/controlnet-openpose-sdxl-1.0",
    "destitech/controlnet-inpaint-dreamer-sdxl",
    "diffusers/controlnet-canny-sdxl-1.0",
    "diffusers/controlnet-canny-sdxl-1.0-mid",
    "diffusers/controlnet-canny-sdxl-1.0-small",
    "diffusers/controlnet-depth-sdxl-1.0",
    "diffusers/controlnet-depth-sdxl-1.0-mid",
    "diffusers/controlnet-depth-sdxl-1.0-small",
]

LOCAL_FILES_ONLY_HELP = """
When loading diffusion models, using local files only, not connect to HuggingFace server.
"""

DEFAULT_MODEL_DIR = os.getenv(
    "XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")
)
MODEL_DIR_HELP = f"""
Model download directory (by setting XDG_CACHE_HOME environment variable), by default model download to {DEFAULT_MODEL_DIR}
"""

OUTPUT_DIR_HELP = """
Result images will be saved to output directory automatically.
"""

INPUT_HELP = """
If input is image, it will be loaded by default.
If input is directory, you can browse and select image in file manager.
"""

GUI_HELP = """
Launch Lama Cleaner as desktop app
"""

QUALITY_HELP = """
Quality of image encoding, 0-100. Default is 95, higher quality will generate larger file size.
"""


class Choices(str, Enum):
    @classmethod
    def values(cls):
        return [member.value for member in cls]


class RealESRGANModel(Choices):
    realesr_general_x4v3 = "realesr-general-x4v3"
    RealESRGAN_x4plus = "RealESRGAN_x4plus"
    RealESRGAN_x4plus_anime_6B = "RealESRGAN_x4plus_anime_6B"


class Device(Choices):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class InteractiveSegModel(Choices):
    vit_b = "vit_b"
    vit_l = "vit_l"
    vit_h = "vit_h"
    mobile_sam = "mobile_sam"


INTERACTIVE_SEG_HELP = "Enable interactive segmentation using Segment Anything."
INTERACTIVE_SEG_MODEL_HELP = "Model size: vit_b < vit_l < vit_h. Bigger model size means better segmentation but slower speed."
REMOVE_BG_HELP = "Enable remove background. Always run on CPU"
ANIMESEG_HELP = "Enable anime segmentation. Always run on CPU"
REALESRGAN_HELP = "Enable realesrgan super resolution"
GFPGAN_HELP = (
    "Enable GFPGAN face restore. To enhance background, use with --enable-realesrgan"
)
RESTOREFORMER_HELP = "Enable RestoreFormer face restore. To enhance background, use with --enable-realesrgan"
GIF_HELP = "Enable GIF plugin. Make GIF to compare original and cleaned image"


class Config(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    model: str = DEFAULT_MODEL
    sd_local_model_path: str = None
    device: str = DEFAULT_DEVICE
    gui: bool = False
    no_gui_auto_close: bool = False
    no_half: bool = False
    cpu_offload: bool = False
    disable_nsfw: bool = False
    sd_cpu_textencoder: bool = False
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
    realesrgan_model: str = RealESRGANModel.realesr_general_x4v3.value
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
