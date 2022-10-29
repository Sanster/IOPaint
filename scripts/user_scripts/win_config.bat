@echo off

set PATH=C:\Windows\System32;%PATH%

@call installer\Scripts\activate.bat

@call conda-unpack

@call conda install -c pytorch -c conda-forge -y pytorch=1.12.1=py3.10_cuda11.6_cudnn8_0

@call pip3 install -U lama-cleaner

@call invoke config
