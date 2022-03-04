# Lama-cleaner: Image inpainting tool powered by SOTA AI model

https://user-images.githubusercontent.com/3998421/153323093-b664bb68-2928-480b-b59b-7c1ee24a4507.mp4

- [x] Support multiple model architectures
    1. [LaMa](https://github.com/saic-mdal/lama)
    1. [LDM](https://github.com/CompVis/latent-diffusion)
- [x] High resolution support
- [x] Multi stroke support. Press and hold the `cmd/ctrl` key to enable multi stroke mode.
- [x] Zoom & Pan
- [ ] Keep image EXIF data

## Quick Start

Install requirements: `pip3 install -r requirements.txt`

### Start server with LaMa model

```bash
python3 main.py --device=cuda --port=8080 --model=lama
```

### Start server with LDM model

```bash
python3 main.py --device=cuda --port=8080 --model=ldm --ldm-steps=50
```

`--ldm-steps`: The larger the value, the better the result, but it will be more time-consuming

Diffusion model is **MUCH MORE** slower than GANs(1080x720 image takes 8s on 3090), but it's possible to get better
results than LaMa.

Blogs about diffusion models:

- https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- https://yang-song.github.io/blog/2021/score/

## Development

Only needed if you plan to modify the frontend and recompile yourself.

### Fronted

Frontend code are modified from [cleanup.pictures](https://github.com/initml/cleanup.pictures), You can experience their
great online services [here](https://cleanup.pictures/).

- Install dependencies:`cd lama_cleaner/app/ && yarn`
- Start development server: `yarn dev`
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
