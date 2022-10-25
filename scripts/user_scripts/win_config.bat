@echo off

set PATH=C:\Windows\System32;%PATH%

@call installer\Scripts\activate.bat

@call conda-unpack

@call pip3 install -U torch==1.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
@call pip3 install -U lama-cleaner

@call invoke config
