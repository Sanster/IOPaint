@echo off

set PATH=C:\Windows\System32;%PATH%

@call installer\Scripts\activate.bat

@call conda-unpack

:CHOICE_CUDA
echo DO YOU WANT TO INSTALL WITH CUDA? (y/n)
set /p Input=Enter y or n:
if "%Input%"=="y" (
   @call conda install -c pytorch -c conda-forge -y pytorch=1.12.1=py3.10_cuda11.6_cudnn8_0
) else if "%Input%"=="n" (
   @call conda install -c pytorch -c conda-forge -y pytorch=1.12.1=py3.10_cpu_0
) else (goto CHOICE_CUDA)

@call pip3 install -U lama-cleaner

@call invoke config
