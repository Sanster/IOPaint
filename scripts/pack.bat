@echo off

set "PYTHONNOUSERSITE=1"

SET BUILD_DIST=lama-cleaner
SET BUILD_ENV=installer
SET USER_SCRIPTS=user_scripts


echo Creating a distributable package..
@call conda env create --prefix %BUILD_ENV% -f environment.yaml

echo Finish creating environment
@call conda activate .\%BUILD_ENV%
@call conda install -c conda-forge -y conda-pack

@call conda pack --n-threads -1 --prefix %BUILD_ENV% --format tar

mkdir %BUILD_DIST%\%BUILD_ENV%

echo "Copy user scripts file %USER_SCRIPTS%"
copy  %USER_SCRIPTS%\* %BUILD_DIST%

cd %BUILD_DIST%
@call tar -xf ..\%BUILD_ENV%.tar -C %BUILD_ENV%

cd ..
@call conda deactivate
rmdir /s /q %BUILD_ENV%
del  %BUILD_ENV%.tar
