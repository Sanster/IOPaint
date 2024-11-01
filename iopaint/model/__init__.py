from .anytext.anytext_model import AnyText
from .controlnet import ControlNet
from .fcf import FcF
from .instruct_pix2pix import InstructPix2Pix
from .kandinsky import Kandinsky22
from .lama import LaMa, AnimeLaMa
from .ldm import LDM
from .manga import Manga
from .mat import MAT
from .mi_gan import MIGAN
from .opencv2 import OpenCV2
from .paint_by_example import PaintByExample
from .power_paint.power_paint import PowerPaint
from .sd import SD15, SD2, Anything4, RealisticVision14, SD
from .sdxl import SDXL
from .zits import ZITS

models = {
    LaMa.name: LaMa,
    AnimeLaMa.name: AnimeLaMa,
    LDM.name: LDM,
    ZITS.name: ZITS,
    MAT.name: MAT,
    FcF.name: FcF,
    OpenCV2.name: OpenCV2,
    Manga.name: Manga,
    MIGAN.name: MIGAN,
    SD15.name: SD15,
    Anything4.name: Anything4,
    RealisticVision14.name: RealisticVision14,
    SD2.name: SD2,
    PaintByExample.name: PaintByExample,
    InstructPix2Pix.name: InstructPix2Pix,
    Kandinsky22.name: Kandinsky22,
    SDXL.name: SDXL,
    PowerPaint.name: PowerPaint,
    AnyText.name: AnyText,
}
