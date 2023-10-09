FROM python:3.10.11-slim-buster

ARG version

RUN apt-get update && apt-get install --no-install-recommends -y \
    software-properties-common \
    libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
    curl gcc build-essential && \
    apt-get clean \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir lama-cleaner==$version

RUN lama-cleaner --install-plugins-package
ENV LD_PRELOAD=/usr/local/lib/python3.10/site-packages/skimage/_shared/../../scikit_image.libs/libgomp-d22c30c5.so.1.0.0

EXPOSE 8080

CMD ["bash"]
