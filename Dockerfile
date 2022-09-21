#
# Lama Cleaner Dockerfile
# @author Loreto Parisi (loretoparisi at gmail dot com)
#

FROM python:3.7.4-slim-buster

LABEL maintainer Loreto Parisi loretoparisi@gmail.com

ENV CACHE_DIR=/app/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
    curl

COPY . .

# download LaMa model
RUN curl -o $CACHE_DIR/hub/checkpoints/big-lama.pt --create-dirs -LJO https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt

# python requirements
RUN pip install -r requirements.txt

CMD ["bash"]
