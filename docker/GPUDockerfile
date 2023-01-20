FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
    curl python3-pip

RUN pip3 install --upgrade pip && \
    pip3 install torch==1.13.1 xformers==0.0.16rc425

ARG version 

RUN pip3 install lama-cleaner==$version

EXPOSE 8080

CMD ["bash"]
