#
# Lama Cleaner Dockerfile
# @author Loreto Parisi (loretoparisi at gmail dot com)
#

FROM python:3.7.4-slim-buster

LABEL maintainer Loreto Parisi loretoparisi@gmail.com

WORKDIR app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
    curl \
    npm

# python requirements
COPY . .
COPY requirements.txt /etc/tmp/requirements.txt
RUN pip install -r /etc/tmp/requirements.txt

# nodejs
RUN npm install n -g && \
    n lts
# yarn
RUN npm install -g yarn

# webapp
RUN cd lama_cleaner/app/ && \
    yarn && \
    yarn build

EXPOSE 8080

CMD ["bash"]