#!/usr/bin/env bash
set -e

pushd ./lama_cleaner/app
yarn run build
popd

rm -r -f dist
python3 setup.py sdist bdist_wheel
twine upload dist/*
