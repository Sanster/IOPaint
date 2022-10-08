#!/usr/bin/env bash
set -e

GIT_TAG=`git describe --tags --abbrev=0`
IMAGE_DESC="Image inpainting tool powered by SOTA AI Model" 
GIT_REPO="https://github.com/Sanster/lama-cleaner"

echo "Building cpu docker image..."

docker buildx build \
--file ./docker/CPUDockerfile \
--label org.opencontainers.image.title=lama-cleaner \
--label org.opencontainers.image.description="$IMAGE_DESC" \
--label org.opencontainers.image.url=$GIT_REPO \
--label org.opencontainers.image.source=$GIT_REPO \
--label org.opencontainers.image.version=$GIT_TAG \
--tag lama-cleaner:cpu-$GIT_TAG .


echo "Building NVIDIA GPU docker image..."

docker buildx build \
--file ./docker/GPUDockerfile \
--label org.opencontainers.image.title=lama-cleaner \
--label org.opencontainers.image.description="$IMAGE_DESC" \
--label org.opencontainers.image.url=$GIT_REPO \
--label org.opencontainers.image.source=$GIT_REPO \
--label org.opencontainers.image.version=$GIT_TAG \
--tag lama-cleaner:gpu-$GIT_TAG .
