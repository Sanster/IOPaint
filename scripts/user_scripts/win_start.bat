@echo off

set PATH=C:\Windows\System32;%PATH%

@call installer\Scripts\activate.bat

set MODEL_DIR=
set COMMANDLINE_ARGS=
@call invoke start

PAUSE