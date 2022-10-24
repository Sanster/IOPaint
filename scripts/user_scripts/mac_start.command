#!/bin/bash

set -e

cd "$(dirname "$0")"
echo `pwd`

source ./installer/bin/activate
conda-unpack

invoke start
