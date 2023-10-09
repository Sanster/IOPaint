FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

ARG version 

RUN apt-get update && apt-get install --no-install-recommends -y \
    software-properties-common \
    libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
    curl python3-pip && \
    apt-get clean \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==1.13.1 xformers==0.0.16rc425 && \
    pip3 install --no-cache-dir lama-cleaner==$version

EXPOSE 8080

CMD ["bash"]
