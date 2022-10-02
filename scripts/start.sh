#!/bin/bash

source installer/bin/activate

conda-unpack

conda --version
git --version

echo Using `which pip3`
pip3 install lama-cleaner

# TODO: add model input prompt
lama-cleaner --device cpu --model lama
