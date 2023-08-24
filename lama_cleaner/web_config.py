import json
import os
from datetime import datetime

import gradio as gr
from loguru import logger

from lama_cleaner.const import *

_config_file = None


def save_config(
    host,
    port,
    model,
    sd_local_model_path,
    sd_controlnet,
    sd_controlnet_method,
    device,
    gui,
    no_gui_auto_close,
    no_half,
    cpu_offload,
    disable_nsfw,
    sd_cpu_textencoder,
    enable_xformers,
    local_files_only,
    model_dir,
    input,
    output_dir,
    quality,
    enable_interactive_seg,
    interactive_seg_model,
    interactive_seg_device,
    enable_remove_bg,
    enable_anime_seg,
    enable_realesrgan,
    realesrgan_device,
    realesrgan_model,
    enable_gfpgan,
    gfpgan_device,
    enable_restoreformer,
    restoreformer_device,
    enable_gif,
):
    config = Config(**locals())
    print(config)
    if config.input and not os.path.exists(config.input):
        return "[Error] Input file or directory does not exist"

    current_time = datetime.now().strftime("%H:%M:%S")
    msg = f"[{current_time}] Successful save config to: {os.path.abspath(_config_file)}"
    logger.info(msg)
    try:
        with open(_config_file, "w", encoding="utf-8") as f:
            json.dump(config.dict(), f, indent=4, ensure_ascii=False)
    except Exception as e:
        return f"Save failed: {str(e)}"
    return msg


def close_server(*args):
    # TODO: make close both browser and server works
    import os, signal

    pid = os.getpid()
    os.kill(pid, signal.SIGUSR1)


def main(config_file: str):
    global _config_file
    _config_file = config_file

    init_config = load_config(config_file)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                save_btn = gr.Button(value="Save configurations")
                message = gr.HTML()

        with gr.Tabs():
            with gr.Tab("Common"):
                with gr.Row():
                    host = gr.Textbox(init_config.host, label="Host")
                    port = gr.Number(init_config.port, label="Port", precision=0)

                model = gr.Radio(
                    AVAILABLE_MODELS, label="Model", value=init_config.model
                )
                device = gr.Radio(
                    AVAILABLE_DEVICES, label="Device", value=init_config.device
                )
                quality = gr.Slider(
                    value=95,
                    label=f"Image Quality ({QUALITY_HELP})",
                    minimum=75,
                    maximum=100,
                    step=1,
                )

                with gr.Column():
                    gui = gr.Checkbox(init_config.gui, label=f"{GUI_HELP}")
                    no_gui_auto_close = gr.Checkbox(
                        init_config.no_gui_auto_close, label=f"{NO_GUI_AUTO_CLOSE_HELP}"
                    )

                with gr.Column():
                    model_dir = gr.Textbox(
                        init_config.model_dir, label=f"{MODEL_DIR_HELP}"
                    )
                    input = gr.Textbox(
                        init_config.input,
                        label=f"Input file or directory. {INPUT_HELP}",
                    )
                    output_dir = gr.Textbox(
                        init_config.output_dir,
                        label=f"Output directory. {OUTPUT_DIR_HELP}",
                    )

            with gr.Tab("Plugins"):
                enable_interactive_seg = gr.Checkbox(
                    init_config.enable_interactive_seg, label=INTERACTIVE_SEG_HELP
                )
                interactive_seg_model = gr.Radio(
                    AVAILABLE_INTERACTIVE_SEG_MODELS,
                    label=f"Segment Anything models. {INTERACTIVE_SEG_MODEL_HELP}",
                    value=init_config.interactive_seg_model,
                )
                interactive_seg_device = gr.Radio(
                    AVAILABLE_INTERACTIVE_SEG_DEVICES,
                    label="Segment Anything Device",
                    value=init_config.interactive_seg_device,
                )
                with gr.Row():
                    enable_remove_bg = gr.Checkbox(
                        init_config.enable_remove_bg, label=REMOVE_BG_HELP
                    )
                with gr.Row():
                    enable_anime_seg = gr.Checkbox(
                        init_config.enable_anime_seg, label=ANIMESEG_HELP
                    )

                with gr.Row():
                    enable_realesrgan = gr.Checkbox(
                        init_config.enable_realesrgan, label=REALESRGAN_HELP
                    )
                    realesrgan_device = gr.Radio(
                        REALESRGAN_AVAILABLE_DEVICES,
                        label="RealESRGAN Device",
                        value=init_config.realesrgan_device,
                    )
                    realesrgan_model = gr.Radio(
                        RealESRGANModelNameList,
                        label="RealESRGAN model",
                        value=init_config.realesrgan_model,
                    )
                with gr.Row():
                    enable_gfpgan = gr.Checkbox(
                        init_config.enable_gfpgan, label=GFPGAN_HELP
                    )
                    gfpgan_device = gr.Radio(
                        GFPGAN_AVAILABLE_DEVICES,
                        label="GFPGAN Device",
                        value=init_config.gfpgan_device,
                    )
                with gr.Row():
                    enable_restoreformer = gr.Checkbox(
                        init_config.enable_restoreformer, label=RESTOREFORMER_HELP
                    )
                    restoreformer_device = gr.Radio(
                        RESTOREFORMER_AVAILABLE_DEVICES,
                        label="RestoreFormer Device",
                        value=init_config.restoreformer_device,
                    )
                enable_gif = gr.Checkbox(init_config.enable_gif, label=GIF_HELP)

            with gr.Tab("Diffusion Model"):
                sd_local_model_path = gr.Textbox(
                    init_config.sd_local_model_path, label=f"{SD_LOCAL_MODEL_HELP}"
                )
                sd_controlnet = gr.Checkbox(
                    init_config.sd_controlnet, label=f"{SD_CONTROLNET_HELP}"
                )
                sd_controlnet_method = gr.Radio(
                    SD_CONTROLNET_CHOICES,
                    lable="ControlNet method",
                    value=init_config.sd_controlnet_method,
                )
                no_half = gr.Checkbox(init_config.no_half, label=f"{NO_HALF_HELP}")
                cpu_offload = gr.Checkbox(
                    init_config.cpu_offload, label=f"{CPU_OFFLOAD_HELP}"
                )
                sd_cpu_textencoder = gr.Checkbox(
                    init_config.sd_cpu_textencoder, label=f"{SD_CPU_TEXTENCODER_HELP}"
                )
                disable_nsfw = gr.Checkbox(
                    init_config.disable_nsfw, label=f"{DISABLE_NSFW_HELP}"
                )
                enable_xformers = gr.Checkbox(
                    init_config.enable_xformers, label=f"{ENABLE_XFORMERS_HELP}"
                )
                local_files_only = gr.Checkbox(
                    init_config.local_files_only, label=f"{LOCAL_FILES_ONLY_HELP}"
                )

        save_btn.click(
            save_config,
            [
                host,
                port,
                model,
                sd_local_model_path,
                sd_controlnet,
                sd_controlnet_method,
                device,
                gui,
                no_gui_auto_close,
                no_half,
                cpu_offload,
                disable_nsfw,
                sd_cpu_textencoder,
                enable_xformers,
                local_files_only,
                model_dir,
                input,
                output_dir,
                quality,
                enable_interactive_seg,
                interactive_seg_model,
                interactive_seg_device,
                enable_remove_bg,
                enable_anime_seg,
                enable_realesrgan,
                realesrgan_device,
                realesrgan_model,
                enable_gfpgan,
                gfpgan_device,
                enable_restoreformer,
                restoreformer_device,
                enable_gif,
            ],
            message,
        )
    demo.launch(inbrowser=True, show_api=False)
