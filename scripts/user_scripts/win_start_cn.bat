@echo off

set PATH=C:\Windows\System32;%PATH%

@call installer\Scripts\activate.bat

@call set HF_ENDPOINT=https://hf-mirror.com
@call iopaint start --config %0\..\installer_config.json

PAUSE
