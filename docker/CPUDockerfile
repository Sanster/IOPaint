FROM python:3.10.11-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
    curl gcc build-essential

RUN pip install --upgrade pip && \
    pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cpu

ARG version

RUN pip install lama-cleaner==$version
RUN lama-cleaner --install-plugins-package
ENV LD_PRELOAD=/usr/local/lib/python3.10/site-packages/skimage/_shared/../../scikit_image.libs/libgomp-d22c30c5.so.1.0.0

EXPOSE 8080

CMD ["bash"]
