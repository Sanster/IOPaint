from loguru import logger

from .interactive_seg import InteractiveSeg
from .remove_bg import RemoveBG
from .realesrgan import RealESRGANUpscaler
from .gfpgan_plugin import GFPGANPlugin
from .restoreformer import RestoreFormerPlugin
from .anime_seg import AnimeSeg
from ..const import InteractiveSegModel, Device


def build_plugins(
    global_config,
    enable_interactive_seg: bool,
    interactive_seg_model: InteractiveSegModel,
    interactive_seg_device: Device,
    enable_remove_bg: bool,
    enable_anime_seg: bool,
    enable_realesrgan: bool,
    realesrgan_device: Device,
    realesrgan_model: str,
    enable_gfpgan: bool,
    gfpgan_device: Device,
    enable_restoreformer: bool,
    restoreformer_device: Device,
    no_half: bool,
):
    if enable_interactive_seg:
        logger.info(f"Initialize {InteractiveSeg.name} plugin")
        global_config.plugins[InteractiveSeg.name] = InteractiveSeg(
            interactive_seg_model, interactive_seg_device
        )

    if enable_remove_bg:
        logger.info(f"Initialize {RemoveBG.name} plugin")
        global_config.plugins[RemoveBG.name] = RemoveBG()

    if enable_anime_seg:
        logger.info(f"Initialize {AnimeSeg.name} plugin")
        global_config.plugins[AnimeSeg.name] = AnimeSeg()

    if enable_realesrgan:
        logger.info(
            f"Initialize {RealESRGANUpscaler.name} plugin: {realesrgan_model}, {realesrgan_device}"
        )
        global_config.plugins[RealESRGANUpscaler.name] = RealESRGANUpscaler(
            realesrgan_model,
            realesrgan_device,
            no_half=no_half,
        )

    if enable_gfpgan:
        logger.info(f"Initialize {GFPGANPlugin.name} plugin")
        if enable_realesrgan:
            logger.info("Use realesrgan as GFPGAN background upscaler")
        else:
            logger.info(
                f"GFPGAN no background upscaler, use --enable-realesrgan to enable it"
            )
        global_config.plugins[GFPGANPlugin.name] = GFPGANPlugin(
            gfpgan_device,
            upscaler=global_config.plugins.get(RealESRGANUpscaler.name, None),
        )

    if enable_restoreformer:
        logger.info(f"Initialize {RestoreFormerPlugin.name} plugin")
        global_config.plugins[RestoreFormerPlugin.name] = RestoreFormerPlugin(
            restoreformer_device,
            upscaler=global_config.plugins.get(RealESRGANUpscaler.name, None),
        )
