<h1 align="center">IOPaint</h1>
<p align="center">A free and open-source inpainting & outpainting tool powered by SOTA AI model.</p>

<p align="center">
  <a href="https://github.com/Sanster/IOPaint">
    <img alt="total download" src="https://pepy.tech/badge/iopaint" />
  </a>
  <a href="https://pypi.org/project/iopaint">
    <img alt="version" src="https://img.shields.io/pypi/v/iopaint" />
  </a>
  <a href="">
    <img alt="python version" src="https://img.shields.io/pypi/pyversions/iopaint" />
  </a>
  <a href="https://huggingface.co/spaces/Sanster/iopaint-lama">
    <img alt="HuggingFace Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces-blue" />
  </a>
  <a href="https://colab.research.google.com/drive/1TKVlDZiE3MIZnAUMpv2t_S4hLr6TUY1d?usp=sharing">
    <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" />
  </a>
</p>

|Erase([LaMa](https://www.iopaint.com/models/erase/lama))|Replace Object([PowerPaint](https://www.iopaint.com/models/diffusion/powerpaint))|
|-----|----|
|<video src="https://github.com/Sanster/IOPaint/assets/3998421/264bc27c-0abd-4d8b-bb1e-0078ab264c4a">  | <video src="https://github.com/Sanster/IOPaint/assets/3998421/1de5c288-e0e1-4f32-926d-796df0655846">|

|Draw Text([AnyText](https://www.iopaint.com/models/diffusion/anytext))|Out-painting([PowerPaint](https://www.iopaint.com/models/diffusion/powerpaint))|
|---------|-----------|
|<video src="https://github.com/Sanster/IOPaint/assets/3998421/ffd4eda4-f7d4-4693-93d8-d2cd5aa7c6d6">|<video src="https://github.com/Sanster/IOPaint/assets/3998421/c4af8aef-8c29-49e0-96eb-0aae2f768da2">|


## Features

- Completely free and open-source, fully self-hosted, support CPU & GPU & Apple Silicon
- [Windows 1-Click Installer](https://www.iopaint.com/install/windows_1click_installer)
- [OptiClean](https://apps.apple.com/ca/app/opticlean/id6452387177): macOS & iOS App for object erase
- Supports various AI [models](https://www.iopaint.com/models) to perform erase, inpainting or outpainting task.
  - [Erase models](https://www.iopaint.com/models#erase-models): These models can be used to remove unwanted object, defect, watermarks, people from image.
  - Diffusion models: These models can be used to replace objects or perform outpainting. Some popular used models include:
    - [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)
    - [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
    - [andregn/Realistic_Vision_V3.0-inpainting](https://huggingface.co/andregn/Realistic_Vision_V3.0-inpainting)
    - [Lykon/dreamshaper-8-inpainting](https://huggingface.co/Lykon/dreamshaper-8-inpainting)
    - [Sanster/anything-4.0-inpainting](https://huggingface.co/Sanster/anything-4.0-inpainting)
    - [BrushNet](https://www.iopaint.com/models/diffusion/brushnet)
    - [PowerPaintV2](https://www.iopaint.com/models/diffusion/powerpaint_v2)
    - [Sanster/AnyText](https://huggingface.co/Sanster/AnyText)
    - [Fantasy-Studio/Paint-by-Example](https://huggingface.co/Fantasy-Studio/Paint-by-Example)

- [Plugins](https://www.iopaint.com/plugins):
  - [Segment Anything](https://iopaint.com/plugins/interactive_seg): Accurate and fast Interactive Object Segmentation
  - [RemoveBG](https://iopaint.com/plugins/rembg): Remove image background or generate masks for foreground objects
  - [Anime Segmentation](https://iopaint.com/plugins/anime_seg): Similar to RemoveBG, the model is specifically trained for anime images.
  - [RealESRGAN](https://iopaint.com/plugins/RealESRGAN): Super Resolution
  - [GFPGAN](https://iopaint.com/plugins/GFPGAN): Face Restoration
  - [RestoreFormer](https://iopaint.com/plugins/RestoreFormer): Face Restoration
- [FileManager](https://iopaint.com/file_manager): Browse your pictures conveniently and save them directly to the output directory.


## Quick Start

### Start webui

IOPaint provides a convenient webui for using the latest AI models to edit your images.
You can install and start IOPaint easily by running following command:

```bash
# In order to use GPU, install cuda version of pytorch first.
# pip3 install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
# AMD GPU users, please utilize the following command, only works on linux, as pytorch is not yet supported on Windows with ROCm.
# pip3 install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/rocm5.6

pip3 install iopaint
iopaint start --model=lama --device=cpu --port=8080
```

That's it, you can start using IOPaint by visiting http://localhost:8080 in your web browser.

All models will be downloaded automatically at startup. If you want to change the download directory, you can add `--model-dir`. More documentation can be found [here](https://www.iopaint.com/install/download_model)

You can see other supported models at [here](https://www.iopaint.com/models) and how to use local sd ckpt/safetensors file at [here](https://www.iopaint.com/models#load-ckptsafetensors).

### Plugins

You can specify which plugins to use when starting the service, and you can view the commands to enable plugins by using `iopaint start --help`. 

More demonstrations of the Plugin can be seen [here](https://www.iopaint.com/plugins)

```bash
iopaint start --enable-interactive-seg --interactive-seg-device=cuda
```

### Batch processing

You can also use IOPaint in the command line to batch process images:

```bash
iopaint run --model=lama --device=cpu \
--image=/path/to/image_folder \
--mask=/path/to/mask_folder \
--output=output_dir
```

`--image` is the folder containing input images, `--mask` is the folder containing corresponding mask images.
When `--mask` is a path to a mask file, all images will be processed using this mask.

You can see more information about the available models and plugins supported by IOPaint below.

## Development

Install [nodejs](https://nodejs.org/en), then install the frontend dependencies.

```bash
git clone https://github.com/Sanster/IOPaint.git
cd IOPaint/web_app
npm install
npm run build
cp -r dist/ ../iopaint/web_app
```

Create a `.env.local` file in `web_app` and fill in the backend IP and port.
```
VITE_BACKEND=http://127.0.0.1:8080
```

Start front-end development environment
```bash
npm run dev
```

Install back-end requirements and start backend service
```bash
pip install -r requirements.txt
python3 main.py start --model lama --port 8080
```

Then you can visit `http://localhost:5173/` for development.
The frontend code will automatically update after being modified,
but the backend needs to restart the service after modifying the python code.
