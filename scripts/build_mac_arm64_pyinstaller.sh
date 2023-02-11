#!/bin/bash

set -e

pyinstaller main.spec --distpath ./lama-cleaner-mac-arm64/dist
cp ./user_scripts/mac_*.command ./lama-cleaner-mac-arm64/