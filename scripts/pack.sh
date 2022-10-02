#!/bin/bash

set -e

export PYTHONNOUSERSITE=1

BUILD_DIST=lama-cleaner
BUILD_ENV=installer

echo "Creating a distributable package.."

source ~/miniconda3/etc/profile.d/conda.sh

conda install -c conda-forge -y conda-pack

conda env create --prefix $BUILD_ENV -f environment.yaml
conda activate ./$BUILD_ENV

conda pack --n-threads -1 --prefix $BUILD_ENV --format tar

mkdir -p ${BUILD_DIST}/$BUILD_ENV

echo "Copy project file.."
chmod u+x start.sh
cp start.sh $BUILD_DIST

cd $BUILD_DIST
tar -xf ../${BUILD_ENV}.tar -C $BUILD_ENV

cd ..
rm -rf $BUILD_ENV
rm ${BUILD_ENV}.tar

zip -r $BUILD_DIST.zip $BUILD_DIST

