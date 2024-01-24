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
</p>

<p align="center">
    <img alt="python version" src="https://github.com/Sanster/models/assets/3998421/561b8395-76a2-4c70-ab24-9f6986924c6a" height=600 />
</p>

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

### Batch processing

You can also use IOPaint in the command line to batch process images: 

```bash
iopaint run --model=lama --device=cpu \
--input=/path/to/image_folder \
--mask=/path/to/mask_folder \
--output=output_dir
```

`--input` is the folder containing input images, `--mask` is the folder containing corresponding mask images. 
When `--mask` is a path to a mask file, all images will be processed using this mask. 

You can see more information about the available models and plugins supported by IOPaint below.

## Features

- Completely free and open-source, fully self-hosted, support CPU & GPU & Apple Silicon
- Supports various AI models:
  - Erase models: These models can be used to remove unwanted object, defect, watermarks, people from image.
  - Stable Diffusion models: You can use any Stable Diffusion Inpainting(or normal) models from [Huggingface](https://huggingface.co/models?other=stable-diffusion) in IOPaint. 
Some popular used models include:
    - [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)
    - [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
    - [andregn/Realistic_Vision_V3.0-inpainting](https://huggingface.co/andregn/Realistic_Vision_V3.0-inpainting)
    - [Lykon/dreamshaper-8-inpainting](https://huggingface.co/Lykon/dreamshaper-8-inpainting)
    - [Sanster/anything-4.0-inpainting](https://huggingface.co/Sanster/anything-4.0-inpainting)
    - [Sanster/PowerPaint-V1-stable-diffusion-inpainting](https://huggingface.co/Sanster/PowerPaint-V1-stable-diffusion-inpainting)
  - Other Diffusion models: 
    - [Sanster/AnyText](https://huggingface.co/Sanster/AnyText)
    - [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)
    - [Fantasy-Studio/Paint-by-Example](https://huggingface.co/Fantasy-Studio/Paint-by-Example)
    - [kandinsky-community/kandinsky-2-2-decoder-inpaint](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint) 
- Plugins
  - [Segment Anything](https://iopaint.com/plugins/interactive_seg): Accurate and fast interactive object segmentation
  - [RemoveBG](https://iopaint.com/plugins/rembg): Remove image background or generate masks for foreground objects
  - [Anime Segmentation](https://iopaint.com/plugins/anime_seg): Similar to RemoveBG, the model is specifically trained for anime images.
  - [RealESRGAN](https://iopaint.com/plugins/RealESRGAN): Super Resolution
  - [GFPGAN](https://iopaint.com/plugins/GFPGAN): Face Restoration
  - [RestoreFormer](https://iopaint.com/plugins/RestoreFormer): Face Restoration
- [FileManager](https://iopaint.com/features/file_manager): Browse your pictures conveniently and save them directly to the output directory.
- [Native macOS app](https://opticlean.io/) for erase task
- More features at [IOPaint Docs](https://iopaint.com/)
