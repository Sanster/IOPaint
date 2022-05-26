# Lama-cleaner: Image inpainting tool powered by SOTA AI model

![downloads](https://img.shields.io/pypi/dm/lama-cleaner)
![version](https://img.shields.io/pypi/v/lama-cleaner)
<a href="https://colab.research.google.com/drive/1e3ZkAJxvkK3uzaTGu91N9TvI_Mahs0Wb?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>


https://user-images.githubusercontent.com/3998421/153323093-b664bb68-2928-480b-b59b-7c1ee24a4507.mp4

- [x] Support multiple model architectures
  1. [LaMa](https://github.com/saic-mdal/lama)
  1. [LDM](https://github.com/CompVis/latent-diffusion)
- [x] Support CPU & GPU
- [x] High resolution support
- [x] Run as a desktop APP
- [x] Multi stroke support. Press and hold the `cmd/ctrl` key to enable multi stroke mode.
- [x] Zoom & Pan

## Install

```bash
pip install lama-cleaner

lama-cleaner --device=cpu --port=8080
```

Available commands:

| Name       | Description                                      | Default  |
| ---------- | ------------------------------------------------ | -------- |
| --model    | lama or ldm. See details in **Model Comparison** | lama     |
| --device   | cuda or cpu                                      | cuda     |
| --gui      | Launch lama-cleaner as a desktop application     |          |
| --gui_size | Set the window size for the application          | 1200 900 |
| --input    | Path to image you want to load by default        | None     |
| --port     | Port for flask web server                        | 8080     |
| --debug    | Enable debug mode for flask web server           |          |

## Model Comparison

Diffusion model(ldm) is **MUCH MORE** slower than GANs(lama)(1080x720 image takes 8s on 3090), but it's possible to get better
result, see below example:

| Original Image                                                                                                                            | LaMa                                                                                                                                                   | LDM                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![photo-1583445095369-9c651e7e5d34](https://user-images.githubusercontent.com/3998421/156923525-d6afdec3-7b98-403f-ad20-88ebc6eb8d6d.jpg) | ![photo-1583445095369-9c651e7e5d34_cleanup_lama](https://user-images.githubusercontent.com/3998421/156923620-a40cc066-fd4a-4d85-a29f-6458711d1247.png) | ![photo-1583445095369-9c651e7e5d34_cleanup_ldm](https://user-images.githubusercontent.com/3998421/156923652-0d06c8c8-33ad-4a42-a717-9c99f3268933.png) |

Blogs about diffusion models:

- https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- https://yang-song.github.io/blog/2021/score/

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

## Like My Work?

<a href="https://www.buymeacoffee.com/Sanster"> 
  <img height="50em" src="https://cdn.buymeacoffee.com/buttons/v2/default-blue.png" alt="Sanster" />
</a>
