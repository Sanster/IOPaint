#!/bin/bash

set -e

cd "$(dirname "$0")"
echo `pwd`

CURRENT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

LAMA_CMD=$CURRENT_DIR/dist/main/main
$LAMA_CMD --config-installer --installer-config $CURRENT_DIR/installer_config.json
