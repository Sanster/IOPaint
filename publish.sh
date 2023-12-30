#!/usr/bin/env bash
set -e

pushd ./web_app
npm run build
popd

rm -r -f dist
python3 setup.py sdist bdist_wheel
twine upload dist/*
