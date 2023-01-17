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
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


CONFIG_PATH = "config.json"


class MODEL(str, Enum):
    SD15 = "sd1.5"
    LAMA = "lama"
    PAINT_BY_EXAMPLE = 'paint_by_example'


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
        c.run('pip list | grep "torch\|lama\|diffusers\|opencv\|cuda\|xformers\|accelerate"')
    except:
        pass
    print("-" * 60)


@task(pre=[info])
def config(c, disable_device_choice=False):
    model = Prompt.ask(
        "Choice model", choices=[MODEL.SD15, MODEL.LAMA, MODEL.PAINT_BY_EXAMPLE], default=MODEL.SD15
    )

    if disable_device_choice:
        device = DEVICE.CPU
    else:
        device = Prompt.ask(
            "Choice device", choices=[DEVICE.CUDA, DEVICE.CPU], default=DEVICE.CUDA
        )
        if device == DEVICE.CUDA:
            import torch

            if not torch.cuda.is_available():
                log.warning(
                    "Did not find CUDA device on your computer, fallback to cpu"
                )
                device = DEVICE.CPU

    desktop = Confirm.ask("Start as desktop app?", default=True)

    configs = {
        "model": model,
        "device": device,
        "desktop": desktop,
    }
    log.info(f"Save config to {CONFIG_PATH}")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)
    Confirm.ask("Config finish, you can close this window")


@task(pre=[info])
def start(c):
    if not os.path.exists(CONFIG_PATH):
        Confirm.ask("Config file not exists, please run config scritp first")
        exit()

    log.info(f"Load config from {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        configs = json.load(f)

    model = configs["model"]
    device = configs["device"]
    desktop = configs["desktop"]
    port = find_free_port()
    log.info(f"Using random port: {port}")

    commandline_args = [
        "--model", model,
        "--device", device,
        "--port", port,
    ]

    if desktop:
        commandline_args.extend(["--gui", "--gui-size", "1400", "900"])

    model_dir = os.environ.get('MODEL_DIR', "")
    if model_dir:
        commandline_args.extend(["--model-dir", model_dir])
    
    commandline_args = ' '.join(commandline_args)
    env_commandline_args = os.environ.get('COMMANDLINE_ARGS', "")

    c.run(
        f"lama-cleaner {env_commandline_args} {commandline_args}"
    )

