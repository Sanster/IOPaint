import os
import json
from enum import Enum
import socket
import logging
from contextlib import closing

from invoke import task
from rich import print
from rich.prompt import IntPrompt, Prompt, Confirm
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("lama-cleaner")


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

CONFIG_PATH = "config.json"

class MODEL(str, Enum):
    SD15 = "sd1.5"
    LAMA = 'lama'

class DEVICE(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"

@task
def info(c):
    print("Environment information".center(60, "-"))
    try:
        c.run("git --version")
        c.run("conda --version")
        c.run("which python")
        c.run("python --version")
        c.run("which pip")
        c.run("pip --version")
        c.run("pip list | grep lama")
    except:
        pass
    print("-"*60)

@task(pre=[info])
def config(c, disable_device_choice=False):
    # TODO: 提示选择模型，选择设备，端口，host
    # 如果是 sd 模型，提示接受条款和输入 huggingface token
    model = Prompt.ask("Choice model", choices=[MODEL.SD15, MODEL.LAMA], default=MODEL.SD15)

    hf_access_token = ""
    if model == MODEL.SD15:
        while True:
            hf_access_token = Prompt.ask("Huggingface access token (https://huggingface.co/docs/hub/security-tokens)")
            if hf_access_token == "":
                log.warning("Access token is required to download model")
            else:
                break  

    if disable_device_choice:
        device = DEVICE.CPU
    else:
        device = Prompt.ask("Choice device", choices=[DEVICE.CUDA, DEVICE.CPU], default=DEVICE.CUDA)
        if device == DEVICE.CUDA:
            import torch
            if not torch.cuda.is_available():
                log.warning("Did not find CUDA device on your computer, fallback to cpu")
                device = DEVICE.CPU

    configs = {"model": model, "device": device, "hf_access_token": hf_access_token}
    log.info(f"Save config to {CONFIG_PATH}")
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)
    log.info(f"Config finish, you can close this window.")


@task(pre=[info]) 
def start(c):
    if not os.path.exists(CONFIG_PATH):
        log.info("Config file not exists, please run config.sh first")
        exit()

    log.info(f"Load config from {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        configs = json.load(f)

    model = configs['model']
    device = configs['device']
    hf_access_token = configs['hf_access_token']
    port = find_free_port()
    log.info(f"Using random port: {port}")

    c.run(f"lama-cleaner --model {model} --device {device} --hf_access_token={hf_access_token} --port {port} --gui --gui-size 1400 900")