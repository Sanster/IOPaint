import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_plugins_package():
    install("onnxruntime<=1.19.2")
    install("rembg[cpu]")
