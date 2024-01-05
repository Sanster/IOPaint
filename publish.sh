#!/usr/bin/env bash
set -e

pushd ./web_app
npm run build
popd
rm -r ./iopaint/web_app
cp -r web_app/dist ./iopaint/web_app

rm -r -f dist
python3 setup.py sdist bdist_wheel
