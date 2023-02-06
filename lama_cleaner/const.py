import os

MPS_SUPPORT_MODELS = [
    "instruct_pix2pix",
    "sd1.5",
    "sd2",
    "paint_by_example"
]

DEFAULT_MODEL = "lama"
AVAILABLE_MODELS = [
    "lama",
    "ldm",
    "zits",
    "mat",
    "fcf",
    "sd1.5",
    "cv2",
    "manga",
    "sd2",
    "paint_by_example",
    "instruct_pix2pix",
]

AVAILABLE_DEVICES = ["cuda", "cpu", "mps"]
DEFAULT_DEVICE = 'cuda'

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

LOCAL_FILES_ONLY_HELP = """
Use local files only, not connect to Hugging Face server. (sd/paint_by_example)
"""

ENABLE_XFORMERS_HELP = """
Enable xFormers optimizations. Requires xformers package has been installed. See: https://github.com/facebookresearch/xformers (sd/paint_by_example)
"""

DEFAULT_MODEL_DIR = os.getenv(
    "XDG_CACHE_HOME",
    os.path.join(os.path.expanduser("~"), ".cache")
)
MODEL_DIR_HELP = """
Model download directory (by setting XDG_CACHE_HOME environment variable), by default model downloaded to ~/.cache
"""

OUTPUT_DIR_HELP = """
Only required when --input is directory. Result images will be saved to output directory automatically.
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
