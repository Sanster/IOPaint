# Lama Cleaner One Click Installer

## Model Description

- **lama**: State of the art image inpainting AI model, useful to remove any unwanted object, defect, people from your pictures.
- **sd1.5**: Stable Diffusion model, text-driven image editing.

## Windows

1. Download [lama-cleaner-win.zip](https://github.com/Sanster/lama-cleaner/releases/download/win_one_click_installer/lama-cleaner-win.zip)
1. Unpack `lama-cleaner-win.zip`
1. Double click `win_config.bat`, follow the guide in the terminal to choice [model](#model-description) and set other configs.
1. Double click `win_start.bat` to start the server.

## Q&A

**How to update the version?**

Rerun `win_config.bat` will install the newest version of lama-cleaner

**Where is model downloaded?**

By default, model will be downloaded to user folder

- stable diffusion model: `C:\Users\your_name\.cache\huggingface`
- lama model: `C:\Users\your_name\.cache\torch`

**How to change the directory of model downloaded?**

Change `win_start.bat` file

```
set TORCH_HOME=your_directory
set HF_HOME=your_directory
@call invoke start
```
