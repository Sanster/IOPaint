@echo off

set PATH=C:\Windows\System32;%PATH%

@call installer\Scripts\activate.bat

@call conda-unpack
@call conda config --set custom_channels.conda-forge https://mirror.sjtu.edu.cn/anaconda/cloud/
@call conda install -y -c conda-forge cudatoolkit=11.7
@call pip install torch torchvision xformers lama-cleaner -i https://download.pytorch.org/whl/cu117 --extra-index-url https://mirrors.ustc.edu.cn/pypi/web/simple
@call lama-cleaner --install-plugins-package

@call lama-cleaner --config-installer --installer-config %0\..\installer_config.json

PAUSE