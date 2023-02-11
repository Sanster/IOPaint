import json
import os
from datetime import datetime

import gradio as gr
from loguru import logger
from pydantic import BaseModel

from lama_cleaner.const import (
    AVAILABLE_MODELS,
    AVAILABLE_DEVICES,
    CPU_OFFLOAD_HELP,
    NO_HALF_HELP,
    DISABLE_NSFW_HELP,
    SD_CPU_TEXTENCODER_HELP,
    LOCAL_FILES_ONLY_HELP,
    ENABLE_XFORMERS_HELP,
    MODEL_DIR_HELP,
    OUTPUT_DIR_HELP,
    INPUT_HELP,
    GUI_HELP,
    DEFAULT_MODEL,
    DEFAULT_DEVICE,
    NO_GUI_AUTO_CLOSE_HELP,
    DEFAULT_MODEL_DIR,
    MPS_SUPPORT_MODELS,
)

_config_file = None


class Config(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    model: str = DEFAULT_MODEL
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


def load_config(installer_config: str):
    if os.path.exists(installer_config):
        with open(installer_config, "r", encoding="utf-8") as f:
            return Config(**json.load(f))
    else:
        return Config()


def save_config(
    host,
    port,
    model,
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
            # with gr.Column(scale=0, min_width=100):
            #     exit_btn = gr.Button(value="Close")
            #     exit_btn.click(close_server)
        with gr.Row():
            host = gr.Textbox(init_config.host, label="Host")
            port = gr.Number(init_config.port, label="Port", precision=0)
        with gr.Row():
            model = gr.Radio(AVAILABLE_MODELS, label="Model", value=init_config.model)
            device = gr.Radio(
                AVAILABLE_DEVICES, label="Device", value=init_config.device
            )
        gui = gr.Checkbox(init_config.gui, label=f"{GUI_HELP}")
        no_gui_auto_close = gr.Checkbox(
            init_config.no_gui_auto_close, label=f"{NO_GUI_AUTO_CLOSE_HELP}"
        )
        no_half = gr.Checkbox(init_config.no_half, label=f"{NO_HALF_HELP}")
        cpu_offload = gr.Checkbox(init_config.cpu_offload, label=f"{CPU_OFFLOAD_HELP}")
        disable_nsfw = gr.Checkbox(
            init_config.disable_nsfw, label=f"{DISABLE_NSFW_HELP}"
        )
        sd_cpu_textencoder = gr.Checkbox(
            init_config.sd_cpu_textencoder, label=f"{SD_CPU_TEXTENCODER_HELP}"
        )
        enable_xformers = gr.Checkbox(
            init_config.enable_xformers, label=f"{ENABLE_XFORMERS_HELP}"
        )
        local_files_only = gr.Checkbox(
            init_config.local_files_only, label=f"{LOCAL_FILES_ONLY_HELP}"
        )
        model_dir = gr.Textbox(init_config.model_dir, label=f"{MODEL_DIR_HELP}")
        input = gr.Textbox(
            init_config.input, label=f"Input file or directory. {INPUT_HELP}"
        )
        output_dir = gr.Textbox(
            init_config.output_dir, label=f"Output directory. {OUTPUT_DIR_HELP}"
        )
        save_btn.click(
            save_config,
            [
                host,
                port,
                model,
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
            ],
            message,
        )
    demo.launch(inbrowser=True, show_api=False)
