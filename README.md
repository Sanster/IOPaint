<h1 align="center">Lama Cleaner</h1>
<p align="center">A free and open-source inpainting tool powered by SOTA AI model.</p>

<p align="center">
  <a href="https://github.com/Sanster/lama-cleaner">
    <img alt="total download" src="https://pepy.tech/badge/lama-cleaner" />
  </a>
  <a href="https://pypi.org/project/lama-cleaner/">
    <img alt="version" src="https://img.shields.io/pypi/v/lama-cleaner" />
  </a>
  <a href="https://colab.research.google.com/drive/1e3ZkAJxvkK3uzaTGu91N9TvI_Mahs0Wb?usp=sharing">
    <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" />
  </a>
  <a href="https://www.buymeacoffee.com/Sanster"> 
    <img height="20px" src="https://cdn.buymeacoffee.com/buttons/v2/default-blue.png" alt="Sanster" />
  </a>
</p>

![img](./assets/dark.jpg)

## Features

- Completely free and open-source
- Fully self-hosted
- Multiple SOTA AI models
  1. [LaMa](https://github.com/saic-mdal/lama)
  1. [LDM](https://github.com/CompVis/latent-diffusion)
  1. [ZITS](https://github.com/DQiaole/ZITS_inpainting)
  1. [MAT](https://github.com/fenglinglwb/MAT)
- Support CPU & GPU
- Various high-resolution image processing [strategy](#high-resolution-strategy)
- Run as a desktop APP

## Usage

| Usage                  | Before                                        | After                                               |
| ---------------------- | --------------------------------------------- | --------------------------------------------------- |
| Remove unwanted things | ![unwant_object2](./assets/unwant_object.jpg) | ![unwant_object2](./assets/unwant_object_clean.jpg) |
| Remove unwanted person | ![unwant_person](./assets/unwant_person.jpg)  | ![unwant_person](./assets/unwant_person_clean.jpg)  |
| Remove Text            | ![text](./assets/unwant_text.jpg)             | ![text](./assets/unwant_text_clean.jpg)             |
| Remove watermark       | ![watermark](./assets/watermark.jpg)          | ![watermark_clean](./assets/watermark_cleanup.jpg)  |
| Fix old photo          | ![oldphoto](./assets/old_photo.jpg)           | ![oldphoto_clean](./assets/old_photo_clean.jpg)     |

## Quick Start

```bash
pip install lama-cleaner

# Model will be downloaded automatically
lama-cleaner --model=lama --device=cpu --port=8080
# Lama Cleaner is now running at http://localhost:8080
```

Available arguments:

| Name       | Description                                                      | Default  |
| ---------- | ---------------------------------------------------------------- | -------- |
| --model    | lama/ldm/zits. See details in [Inpaint Model](#inpainting-model) | lama     |
| --device   | cuda or cpu                                                      | cuda     |
| --port     | Port for backend flask web server                                | 8080     |
| --gui      | Launch lama-cleaner as a desktop application                     |          |
| --gui_size | Set the window size for the application                          | 1200 900 |
| --input    | Path to image you want to load by default                        | None     |
| --debug    | Enable debug mode for flask web server                           |          |

## Inpainting Model

| Model | Description                                                                                                                                                                                                                 | Config                                                                                                                                                                                                                                                                            |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LaMa  | :+1: Generalizes well on high resolutions(~2k)<br/>                                                                                                                                                                         |                                                                                                                                                                                                                                                                                   |
| LDM   | :+1: Possiblablity to get better and more detail result <br/> :+1: The balance of time and quality can be achieved by adjusting `steps` <br/> :neutral_face: Slower than GAN model<br/> :neutral_face: Need more GPU memory | `Steps`: You can get better result with large steps, but it will be more time-consuming <br/> `Sampler`: ddim or [plms](https://arxiv.org/abs/2202.09778). In general plms can get [better results](https://github.com/Sanster/lama-cleaner/releases/tag/0.13.0) with fewer steps |
| ZITS  | :+1: Better holistic structures compared with previous methods <br/> :neutral_face: Wireframe module is **very** slow on CPU                                                                                                | `Wireframe`: Enable edge and line detect                                                                                                                                                                                                                                          |
| MAT   | TODO                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                   |

### LaMa vs LDM

| Original Image                                                                                                                            | LaMa                                                                                                                                                   | LDM                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![photo-1583445095369-9c651e7e5d34](https://user-images.githubusercontent.com/3998421/156923525-d6afdec3-7b98-403f-ad20-88ebc6eb8d6d.jpg) | ![photo-1583445095369-9c651e7e5d34_cleanup_lama](https://user-images.githubusercontent.com/3998421/156923620-a40cc066-fd4a-4d85-a29f-6458711d1247.png) | ![photo-1583445095369-9c651e7e5d34_cleanup_ldm](https://user-images.githubusercontent.com/3998421/156923652-0d06c8c8-33ad-4a42-a717-9c99f3268933.png) |

### LaMa vs ZITS

| Original Image                                                                                                         | ZITS                                                                                                                       | LaMa                                                                                                                       |
| ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| ![zits_original](https://user-images.githubusercontent.com/3998421/180464918-eb13ebfb-8718-461c-9e8b-7f6d8bb7a84f.png) | ![zits_compare_zits](https://user-images.githubusercontent.com/3998421/180464914-4db722c9-047f-48fe-9bb4-916ba09eb5c6.png) | ![zits_compare_lama](https://user-images.githubusercontent.com/3998421/180464903-ffb5f770-4372-4488-ba76-4b4a8c3323f5.png) |

Image is from [ZITS](https://github.com/DQiaole/ZITS_inpainting) paper. I didn't find a good example to show the advantages of ZITS and let me know if you have a good example. There can also be possible problems with my code, if you find them, please let me know too!

## High Resolution Strategy

There are three strategies for handling high-resolution images.

- **Original**: Use the original resolution of the picture, suitable for picture size below 2K.
- **Resize**: Resize the image to a smaller size, then do inpainting on the resized image. The inpainting area will be croped, upsampled and pasted back on the original image to make sure other part of image not loss quality.
- **Crop**: Crop masking area from the original image to do inpainting, and paste the result back. Mainly for performance and memory reasons on high resolution image.

## Download Model Mannually

If you have problems downloading the model automatically when lama-cleaner start,
you can download it manually. By default lama-cleaner will load model from `TORCH_HOME=~/.cache/torch/hub/checkpoints/`,
you can set `TORCH_HOME` to other folder and put the models there.

- Github:
  - [LaMa](https://github.com/Sanster/models/releases/tag/add_big_lama)
  - [LDM](https://github.com/Sanster/models/releases/tag/add_ldm)
  - [ZITS](https://github.com/Sanster/models/releases/tag/add_zits)
- Baidu:
  - https://pan.baidu.com/s/1vUd3BVqIpK6e8N_EA_ZJfw
  - passward: flsu

## Development

Only needed if you plan to modify the frontend and recompile yourself.

### Frontend

Frontend code are modified from [cleanup.pictures](https://github.com/initml/cleanup.pictures), You can experience their
great online services [here](https://cleanup.pictures/).

- Install dependencies:`cd lama_cleaner/app/ && yarn`
- Start development server: `yarn start`
- Build: `yarn build`

## Docker

Run within a Docker container. Set the `CACHE_DIR` to models location path. Optionally add a `-d` option to
the `docker run` command below to run as a daemon.

### Build Docker image

```
docker build -f Dockerfile -t lamacleaner .
```

### Run Docker (cpu)

```
docker run -p 8080:8080 -e CACHE_DIR=/app/models -v  $(pwd)/models:/app/models -v $(pwd):/app --rm lamacleaner python3 main.py --device=cpu --port=8080
```

### Run Docker (gpu)

```
docker run --gpus all -p 8080:8080 -e CACHE_DIR=/app/models -v $(pwd)/models:/app/models -v $(pwd):/app --rm lamacleaner python3 main.py --device=cuda --port=8080
```

Then open [http://localhost:8080](http://localhost:8080)
