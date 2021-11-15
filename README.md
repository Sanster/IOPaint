# Lama-cleaner: Image inpainting tool powered by [LaMa](https://github.com/saic-mdal/lama)

This project is mainly used for selfhosting LaMa model, some interaction improvements may be added later.

## Quick Start

- Install requirements: `pip3 install -r requirements.txt`
- Start server: `python3 main.py --device=cuda --port=8080`

## Development

### Fronted

Frontend code are modified from [cleanup.pictures](https://github.com/initml/cleanup.pictures),
You can experience their great online services [here](https://cleanup.pictures/).

- Install dependencies:`cd lama_cleaner/app/ && yarn`
- Start development server: `yarn dev`
- Build: `yarn build`

## Docker (cpu)
```
docker build -f Dockerfile -t lamacleaner .
docker run -p 8080:8080 -e cache_dir=/app/models -v models:/app/models -v $(pwd):/app --rm lamacleaner python3 main.py --device=cpu --port=8080
```

## Docker (gpu)
```
docker build -f Dockerfile -t lamacleaner .
docker run --gpus all -p 8080:8080 -e cache_dir=/app/models -v models:/app/models -v $(pwd):/app --rm lamacleaner python3 main.py --device=gpu --port=8080
```

Then open [http://localhost:8080](http://localhost:8080)