import random
from enum import Enum
from pathlib import Path
from typing import Optional, Literal, List

from loguru import logger

from iopaint.const import (
    INSTRUCT_PIX2PIX_NAME,
    KANDINSKY22_NAME,
    POWERPAINT_NAME,
    ANYTEXT_NAME,
    SDXL_CONTROLNET_CHOICES,
    SD2_CONTROLNET_CHOICES,
    SD_CONTROLNET_CHOICES,
    SD_BRUSHNET_CHOICES,
    SDXL_BRUSHNET_CHOICES
)
from pydantic import BaseModel, Field, computed_field, model_validator


class ModelType(str, Enum):
    INPAINT = "inpaint"  # LaMa, MAT...
    DIFFUSERS_SD = "diffusers_sd"
    DIFFUSERS_SD_INPAINT = "diffusers_sd_inpaint"
    DIFFUSERS_SDXL = "diffusers_sdxl"
    DIFFUSERS_SDXL_INPAINT = "diffusers_sdxl_inpaint"
    DIFFUSERS_OTHER = "diffusers_other"


class ModelInfo(BaseModel):
    name: str
    path: str
    model_type: ModelType
    is_single_file_diffusers: bool = False

    @computed_field
    @property
    def need_prompt(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ] or self.name in [
            INSTRUCT_PIX2PIX_NAME,
            KANDINSKY22_NAME,
            POWERPAINT_NAME,
            ANYTEXT_NAME,
        ]

    @computed_field
    @property
    def controlnets(self) -> List[str]:
        if self.model_type in [
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ]:
            return SDXL_CONTROLNET_CHOICES
        if self.model_type in [ModelType.DIFFUSERS_SD, ModelType.DIFFUSERS_SD_INPAINT]:
            if "sd2" in self.name.lower():
                return SD2_CONTROLNET_CHOICES
            else:
                return SD_CONTROLNET_CHOICES
        if self.name == POWERPAINT_NAME:
            return SD_CONTROLNET_CHOICES
        return []

    @computed_field
    @property
    def brushnets(self) -> List[str]:
        if self.model_type in [ModelType.DIFFUSERS_SD]:
            return SD_BRUSHNET_CHOICES
        if self.model_type in [ModelType.DIFFUSERS_SDXL]:
            return SDXL_BRUSHNET_CHOICES
        return []

    @computed_field
    @property
    def support_strength(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ] or self.name in [POWERPAINT_NAME, ANYTEXT_NAME]

    @computed_field
    @property
    def support_outpainting(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ] or self.name in [KANDINSKY22_NAME, POWERPAINT_NAME]

    @computed_field
    @property
    def support_lcm_lora(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ]

    @computed_field
    @property
    def support_controlnet(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ]

    @computed_field
    @property
    def support_brushnet(self) -> bool:
        return self.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SDXL,
        ]

    @computed_field
    @property
    def support_powerpaint_v2(self) -> bool:
        return (
            self.model_type
            in [
                ModelType.DIFFUSERS_SD,
            ]
            and self.name != POWERPAINT_NAME
        )


class Choices(str, Enum):
    @classmethod
    def values(cls):
        return [member.value for member in cls]


class RealESRGANModel(Choices):
    realesr_general_x4v3 = "realesr-general-x4v3"
    RealESRGAN_x4plus = "RealESRGAN_x4plus"
    RealESRGAN_x4plus_anime_6B = "RealESRGAN_x4plus_anime_6B"


class RemoveBGModel(Choices):
    briaai_rmbg_1_4 = "briaai/RMBG-1.4"
    briaai_rmbg_2_0 = "briaai/RMBG-2.0"
    # models from https://github.com/danielgatis/rembg
    u2net = "u2net"
    u2netp = "u2netp"
    u2net_human_seg = "u2net_human_seg"
    u2net_cloth_seg = "u2net_cloth_seg"
    silueta = "silueta"
    isnet_general_use = "isnet-general-use"
    birefnet_general = "birefnet-general"
    birefnet_general_lite = "birefnet-general-lite"
    birefnet_portrait = "birefnet-portrait"
    birefnet_dis = "birefnet-dis"
    birefnet_hrsod = "birefnet-hrsod"
    birefnet_cod = "birefnet-cod"
    birefnet_massive = "birefnet-massive"


class Device(Choices):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class InteractiveSegModel(Choices):
    vit_b = "vit_b"
    vit_l = "vit_l"
    vit_h = "vit_h"
    sam_hq_vit_b = "sam_hq_vit_b"
    sam_hq_vit_l = "sam_hq_vit_l"
    sam_hq_vit_h = "sam_hq_vit_h"
    mobile_sam = "mobile_sam"

    sam2_tiny = "sam2_tiny"
    sam2_small = "sam2_small"
    sam2_base = "sam2_base"
    sam2_large = "sam2_large"

    sam2_1_tiny = "sam2_1_tiny"
    sam2_1_small = "sam2_1_small"
    sam2_1_base = "sam2_1_base"
    sam2_1_large = "sam2_1_large"


class PluginInfo(BaseModel):
    name: str
    support_gen_image: bool = False
    support_gen_mask: bool = False


class CV2Flag(str, Enum):
    INPAINT_NS = "INPAINT_NS"
    INPAINT_TELEA = "INPAINT_TELEA"


class HDStrategy(str, Enum):
    # Use original image size
    ORIGINAL = "Original"
    # Resize the longer side of the image to a specific size(hd_strategy_resize_limit),
    # then do inpainting on the resized image. Finally, resize the inpainting result to the original size.
    # The area outside the mask will not lose quality.
    RESIZE = "Resize"
    # Crop masking area(with a margin controlled by hd_strategy_crop_margin) from the original image to do inpainting
    CROP = "Crop"


class LDMSampler(str, Enum):
    ddim = "ddim"
    plms = "plms"


class SDSampler(str, Enum):
    dpm_plus_plus_2m = "DPM++ 2M"
    dpm_plus_plus_2m_karras = "DPM++ 2M Karras"
    dpm_plus_plus_2m_sde = "DPM++ 2M SDE"
    dpm_plus_plus_2m_sde_karras = "DPM++ 2M SDE Karras"
    dpm_plus_plus_sde = "DPM++ SDE"
    dpm_plus_plus_sde_karras = "DPM++ SDE Karras"
    dpm2 = "DPM2"
    dpm2_karras = "DPM2 Karras"
    dpm2_a = "DPM2 a"
    dpm2_a_karras = "DPM2 a Karras"
    euler = "Euler"
    euler_a = "Euler a"
    heun = "Heun"
    lms = "LMS"
    lms_karras = "LMS Karras"

    ddim = "DDIM"
    pndm = "PNDM"
    uni_pc = "UniPC"
    lcm = "LCM"


class PowerPaintTask(Choices):
    text_guided = "text-guided"
    context_aware = "context-aware"
    shape_guided = "shape-guided"
    object_remove = "object-remove"
    outpainting = "outpainting"


class ApiConfig(BaseModel):
    host: str
    port: int
    inbrowser: bool
    model: str
    no_half: bool
    low_mem: bool
    cpu_offload: bool
    disable_nsfw_checker: bool
    local_files_only: bool
    cpu_textencoder: bool
    device: Device
    input: Optional[Path]
    mask_dir: Optional[Path]
    output_dir: Optional[Path]
    quality: int
    enable_interactive_seg: bool
    interactive_seg_model: InteractiveSegModel
    interactive_seg_device: Device
    enable_remove_bg: bool
    remove_bg_device: Device
    remove_bg_model: str
    enable_anime_seg: bool
    enable_realesrgan: bool
    realesrgan_device: Device
    realesrgan_model: RealESRGANModel
    enable_gfpgan: bool
    gfpgan_device: Device
    enable_restoreformer: bool
    restoreformer_device: Device


class InpaintRequest(BaseModel):
    image: Optional[str] = Field(None, description="base64 encoded image")
    mask: Optional[str] = Field(None, description="base64 encoded mask")

    ldm_steps: int = Field(20, description="Steps for ldm model.")
    ldm_sampler: str = Field(LDMSampler.plms, description="Sampler for ldm model.")
    zits_wireframe: bool = Field(True, description="Enable wireframe for zits model.")

    hd_strategy: str = Field(
        HDStrategy.CROP,
        description="Different way to preprocess image, only used by erase models(e.g. lama/mat)",
    )
    hd_strategy_crop_trigger_size: int = Field(
        800,
        description="Crop trigger size for hd_strategy=CROP, if the longer side of the image is larger than this value, use crop strategy",
    )
    hd_strategy_crop_margin: int = Field(
        128, description="Crop margin for hd_strategy=CROP"
    )
    hd_strategy_resize_limit: int = Field(
        1280, description="Resize limit for hd_strategy=RESIZE"
    )

    prompt: str = Field("", description="Prompt for diffusion models.")
    negative_prompt: str = Field(
        "", description="Negative prompt for diffusion models."
    )
    use_croper: bool = Field(
        False, description="Crop image before doing diffusion inpainting"
    )
    croper_x: int = Field(0, description="Crop x for croper")
    croper_y: int = Field(0, description="Crop y for croper")
    croper_height: int = Field(512, description="Crop height for croper")
    croper_width: int = Field(512, description="Crop width for croper")

    use_extender: bool = Field(
        False, description="Extend image before doing sd outpainting"
    )
    extender_x: int = Field(0, description="Extend x for extender")
    extender_y: int = Field(0, description="Extend y for extender")
    extender_height: int = Field(640, description="Extend height for extender")
    extender_width: int = Field(640, description="Extend width for extender")

    sd_scale: float = Field(
        1.0,
        description="Resize the image before doing sd inpainting, the area outside the mask will not lose quality.",
        gt=0.0,
        le=1.0,
    )
    sd_mask_blur: int = Field(
        11,
        description="Blur the edge of mask area. The higher the number the smoother blend with the original image",
    )
    sd_strength: float = Field(
        1.0,
        description="Strength is a measure of how much noise is added to the base image, which influences how similar the output is to the base image. Higher value means more noise and more different from the base image",
        le=1.0,
    )
    sd_steps: int = Field(
        50,
        description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
    )
    sd_guidance_scale: float = Field(
        7.5,
        description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.",
    )
    sd_sampler: str = Field(
        SDSampler.uni_pc, description="Sampler for diffusion model."
    )
    sd_seed: int = Field(
        42,
        description="Seed for diffusion model. -1 mean random seed",
        validate_default=True,
    )
    sd_match_histograms: bool = Field(
        False,
        description="Match histograms between inpainting area and original image.",
    )

    sd_outpainting_softness: float = Field(20.0)
    sd_outpainting_space: float = Field(20.0)

    sd_lcm_lora: bool = Field(
        False,
        description="Enable lcm-lora mode. https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm#texttoimage",
    )

    sd_keep_unmasked_area: bool = Field(
        True, description="Keep unmasked area unchanged"
    )

    cv2_flag: CV2Flag = Field(
        CV2Flag.INPAINT_NS,
        description="Flag for opencv inpainting: https://docs.opencv.org/4.6.0/d7/d8b/group__photo__inpaint.html#gga8002a65f5a3328fbf15df81b842d3c3ca05e763003a805e6c11c673a9f4ba7d07",
    )
    cv2_radius: int = Field(
        4,
        description="Radius of a circular neighborhood of each point inpainted that is considered by the algorithm",
    )

    # Paint by Example
    paint_by_example_example_image: Optional[str] = Field(
        None, description="Base64 encoded example image for paint by example model"
    )

    # InstructPix2Pix
    p2p_image_guidance_scale: float = Field(1.5, description="Image guidance scale")

    # ControlNet
    enable_controlnet: bool = Field(False, description="Enable controlnet")
    controlnet_conditioning_scale: float = Field(
        0.4, description="Conditioning scale", ge=0.0, le=1.0
    )
    controlnet_method: str = Field(
        "lllyasviel/control_v11p_sd15_canny", description="Controlnet method"
    )

    # BrushNet
    enable_brushnet: bool = Field(False, description="Enable brushnet")
    brushnet_method: str = Field(SD_BRUSHNET_CHOICES[0], description="Brushnet method")
    brushnet_conditioning_scale: float = Field(
        1.0, description="brushnet conditioning scale", ge=0.0, le=1.0
    )

    # PowerPaint
    enable_powerpaint_v2: bool = Field(False, description="Enable PowerPaint v2")
    powerpaint_task: PowerPaintTask = Field(
        PowerPaintTask.text_guided, description="PowerPaint task"
    )
    fitting_degree: float = Field(
        1.0,
        description="Control the fitting degree of the generated objects to the mask shape.",
        gt=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_field(cls, values: "InpaintRequest"):
        if values.sd_seed == -1:
            values.sd_seed = random.randint(1, 99999999)
            logger.info(f"Generate random seed: {values.sd_seed}")

        if values.use_extender and values.enable_controlnet:
            logger.info("Extender is enabled, set controlnet_conditioning_scale=0")
            values.controlnet_conditioning_scale = 0

        if values.use_extender:
            logger.info("Extender is enabled, set sd_strength=1")
            values.sd_strength = 1.0

        if values.enable_brushnet:
            logger.info("BrushNet is enabled, set enable_controlnet=False")
            if values.enable_controlnet:
                values.enable_controlnet = False
            if values.sd_lcm_lora:
                logger.info("BrushNet is enabled, set sd_lcm_lora=False")
                values.sd_lcm_lora = False

        if values.enable_controlnet:
            logger.info("ControlNet is enabled, set enable_brushnet=False")
            if values.enable_brushnet:
                values.enable_brushnet = False

        return values


class RunPluginRequest(BaseModel):
    name: str
    image: str = Field(..., description="base64 encoded image")
    clicks: List[List[int]] = Field(
        [], description="Clicks for interactive seg, [[x,y,0/1], [x2,y2,0/1]]"
    )
    scale: float = Field(2.0, description="Scale for upscaling")


MediaTab = Literal["input", "output", "mask"]


class MediasResponse(BaseModel):
    name: str
    height: int
    width: int
    ctime: float
    mtime: float


class GenInfoResponse(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""


class ServerConfigResponse(BaseModel):
    plugins: List[PluginInfo]
    modelInfos: List[ModelInfo]
    removeBGModel: RemoveBGModel
    removeBGModels: List[RemoveBGModel]
    realesrganModel: RealESRGANModel
    realesrganModels: List[RealESRGANModel]
    interactiveSegModel: InteractiveSegModel
    interactiveSegModels: List[InteractiveSegModel]
    enableFileManager: bool
    enableAutoSaving: bool
    enableControlnet: bool
    controlnetMethod: Optional[str]
    disableModelSwitch: bool
    isDesktop: bool
    samplers: List[str]


class SwitchModelRequest(BaseModel):
    name: str


class SwitchPluginModelRequest(BaseModel):
    plugin_name: str
    model_name: str


AdjustMaskOperate = Literal["expand", "shrink", "reverse"]


class AdjustMaskRequest(BaseModel):
    mask: str = Field(
        ..., description="base64 encoded mask. 255 means area to do inpaint"
    )
    operate: AdjustMaskOperate = Field(..., description="expand/shrink/reverse")
    kernel_size: int = Field(5, description="Kernel size for expanding mask")
