import os
from enum import Enum

MPS_SUPPORT_MODELS = [
    "instruct_pix2pix",
    "sd1.5",
    "anything4",
    "realisticVision1.4",
    "sd2",
    "paint_by_example",
    "controlnet"
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
Run Stable Diffusion 1.5 inpainting model with Canny ControlNet control.
"""

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

INTERACTIVE_SEG_HELP = "Enable interactive segmentation. Always run on CPU"
REMOVE_BG_HELP = "Enable remove background. Always run on CPU"
REALESRGAN_HELP = "Enable realesrgan super resolution"
REALESRGAN_AVAILABLE_DEVICES = ["cpu", "cuda", "mps"]
GFPGAN_HELP = (
    "Enable GFPGAN face restore. To enhance background, use with --enable-realesrgan"
)
GFPGAN_AVAILABLE_DEVICES = ["cpu", "cuda", "mps"]
RESTOREFORMER_HELP = "Enable RestoreFormer face restore. To enhance background, use with --enable-realesrgan"
RESTOREFORMER_AVAILABLE_DEVICES = ["cpu", "cuda", "mps"]
GIF_HELP = "Enable GIF plugin. Make GIF to compare original and cleaned image"
