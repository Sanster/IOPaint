import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip3", "install", package])


def install_plugins_package():
    install("rembg")
