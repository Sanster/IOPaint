import os
from typing import List

INSTRUCT_PIX2PIX_NAME = "timbrooks/instruct-pix2pix"
KANDINSKY22_NAME = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
POWERPAINT_NAME = "Sanster/PowerPaint-V1-stable-diffusion-inpainting"
ANYTEXT_NAME = "Sanster/AnyText"

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
DIFFUSION_MODELS = [
    "runwayml/stable-diffusion-inpainting",
    "Uminosachi/realisticVisionV51_v51VAE-inpainting",
    "redstonehero/dreamshaper-inpainting",
    "Sanster/anything-4.0-inpainting",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "Fantasy-Studio/Paint-by-Example",
    POWERPAINT_NAME,
    ANYTEXT_NAME,
]

NO_HALF_HELP = """
Using full precision(fp32) model.
If your diffusion model generate result is always black or green, use this argument.
"""

CPU_OFFLOAD_HELP = """
Offloads diffusion model's weight to CPU RAM, significantly reducing vRAM usage.
"""

LOW_MEM_HELP = "Enable attention slicing and vae tiling to save memory."

DISABLE_NSFW_HELP = """
Disable NSFW checker for diffusion model.
"""

CPU_TEXTENCODER_HELP = """
Run diffusion models text encoder on CPU to reduce vRAM usage.
"""

SD_CONTROLNET_CHOICES: List[str] = [
    "lllyasviel/control_v11p_sd15_canny",
    # "lllyasviel/control_v11p_sd15_seg",
    "lllyasviel/control_v11p_sd15_openpose",
    "lllyasviel/control_v11p_sd15_inpaint",
    "lllyasviel/control_v11f1p_sd15_depth",
]

SD_BRUSHNET_CHOICES: List[str] = [
    "Sanster/brushnet_random_mask",
    "Sanster/brushnet_segmentation_mask",
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

DEFAULT_MODEL_DIR = os.path.abspath(
    os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
)

MODEL_DIR_HELP = f"""
Model download directory (by setting XDG_CACHE_HOME environment variable), by default model download to {DEFAULT_MODEL_DIR}
"""

OUTPUT_DIR_HELP = """
Result images will be saved to output directory automatically.
"""

MASK_DIR_HELP = """
You can view masks in FileManager
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

INTERACTIVE_SEG_HELP = "Enable interactive segmentation using Segment Anything."
INTERACTIVE_SEG_MODEL_HELP = "Model size: mobile_sam < vit_b < vit_l < vit_h. Bigger model size means better segmentation but slower speed."
REMOVE_BG_HELP = "Enable remove background plugin. Always run on CPU"
ANIMESEG_HELP = "Enable anime segmentation plugin. Always run on CPU"
REALESRGAN_HELP = "Enable realesrgan super resolution"
GFPGAN_HELP = "Enable GFPGAN face restore. To also enhance background, use with --enable-realesrgan"
RESTOREFORMER_HELP = "Enable RestoreFormer face restore. To also enhance background, use with --enable-realesrgan"
GIF_HELP = "Enable GIF plugin. Make GIF to compare original and cleaned image"

INBROWSER_HELP = "Automatically launch IOPaint in a new tab on the default browser"
