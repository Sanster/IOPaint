#!/bin/bash

set -e

cd "$(dirname "$0")"
echo `pwd`

source ./installer/bin/activate
conda-unpack

pip3 install -U lama-cleaner

invoke config --disable-device-choice
