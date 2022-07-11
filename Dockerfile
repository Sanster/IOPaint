#
# Lama Cleaner Dockerfile
# @author Reddexx
#

FROM python:3.8.13-slim-bullseye
ENV LAMA_CLEANER_VERSION=0.14.0
LABEL maintainer Reddexx

WORKDIR app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
    curl \
    npm

# python requirements

RUN pip install torch>=1.8.2 opencv-python flask_cors flask==1.1.4 flaskwebgui tqdm pydantic loguru pytest markupsafe==2.0.1

# nodejs
RUN npm install n -g && \
    n lts
# yarn
RUN npm install -g yarn

#Create Directory
RUN mkdir -p /lama_cleaner && cd /lama_cleaner

# webapp
RUN set -x; curl -SL -o lama-cleaner.tar.gz https://github.com/Sanster/lama-cleaner/archive/refs/tags/${LAMA_CLEANER_VERSION}.tar.gz && \
tar xvf lama-cleaner.tar.gz -C /lama_cleaner --strip-components=1 && \
rm lama-cleaner.tar.gz

RUN cd /lama_cleaner/lama_cleaner/app && yarn && yarn build

EXPOSE 8080

CMD ["bash"]
