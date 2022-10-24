@echo off

export "PYTHONNOUSERSITE=1"
SET BUILD_DIST=lama-cleaner
SET BUILD_ENV=installer
SET USER_SCRIPTS=user_scripts
echo "Creating a distributable package.."
source "~\miniconda3\etc\profile.d\conda.sh"
conda "install" "-c" "conda-forge" "-y" "conda-pack"
conda "env" "create" "--prefix" "%BUILD_ENV%" "-f" "environment.yaml"
conda "activate" "%CD%\%BUILD_ENV%"
conda "pack" "--n-threads" "-1" "--prefix" "%BUILD_ENV%" "--format" "tar"
mkdir "-p" "%BUILD_DIST%/%BUILD_ENV%"
echo "Copy user scripts file %USER_SCRIPTS%"
COPY  "%USER_SCRIPTS%/*" "%BUILD_DIST%"
cd "%BUILD_DIST%"
tar "-xf" "%CD%.\%BUILD_ENV%%CD%tar" "-C" "%BUILD_ENV%"
cd "%CD%."
DEL /S "%BUILD_ENV%"
DEL  "%BUILD_ENV%%CD%tar"
echo "zip %BUILD_DIST%%CD%zip"
zip "-q" "-r" "%BUILD_DIST%%CD%zip" "%BUILD_DIST%"