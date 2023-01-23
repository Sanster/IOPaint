@echo off

set PATH=C:\Windows\System32;%PATH%

@call installer\Scripts\activate.bat

@call lama-cleaner --load-installer-config --installer-config %0\..\installer_config.json

PAUSE