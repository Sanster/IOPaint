import os
from io import BytesIO
from pathlib import Path
from typing import List

from PIL import Image, ImageOps, PngImagePlugin
from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse

from ..schema import MediasResponse, MediaTab

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
from .storage_backends import FilesystemStorageBackend
from .utils import aspect_to_string, generate_filename, glob_img


class FileManager:
    def __init__(self, app: FastAPI, input_dir: Path, mask_dir: Path, output_dir: Path):
        self.app = app
        self.input_dir: Path = input_dir
        self.mask_dir: Path = mask_dir
        self.output_dir: Path = output_dir

        self.image_dir_filenames = []
        self.output_dir_filenames = []
        if not self.thumbnail_directory.exists():
            self.thumbnail_directory.mkdir(parents=True)

        # fmt: off
        self.app.add_api_route("/api/v1/medias", self.api_medias, methods=["GET"], response_model=List[MediasResponse])
        self.app.add_api_route("/api/v1/media_file", self.api_media_file, methods=["GET"])
        self.app.add_api_route("/api/v1/media_thumbnail_file", self.api_media_thumbnail_file, methods=["GET"])
        # fmt: on

    def api_medias(self, tab: MediaTab) -> List[MediasResponse]:
        img_dir = self._get_dir(tab)
        return self._media_names(img_dir)

    def api_media_file(self, tab: MediaTab, filename: str) -> FileResponse:
        file_path = self._get_file(tab, filename)
        return FileResponse(file_path, media_type="image/png")

    # tab=${tab}?filename=${filename.name}?width=${width}&height=${height}
    def api_media_thumbnail_file(
        self, tab: MediaTab, filename: str, width: int, height: int
    ) -> FileResponse:
        img_dir = self._get_dir(tab)
        thumb_filename, (width, height) = self.get_thumbnail(
            img_dir, filename, width=width, height=height
        )
        thumbnail_filepath = self.thumbnail_directory / thumb_filename
        return FileResponse(
            thumbnail_filepath,
            headers={
                "X-Width": str(width),
                "X-Height": str(height),
            },
            media_type="image/jpeg",
        )

    def _get_dir(self, tab: MediaTab) -> Path:
        if tab == "input":
            return self.input_dir
        elif tab == "output":
            return self.output_dir
        elif tab == "mask":
            return self.mask_dir
        else:
            raise HTTPException(status_code=422, detail=f"tab not found: {tab}")

    def _get_file(self, tab: MediaTab, filename: str) -> Path:
        file_path = self._get_dir(tab) / filename
        if not file_path.exists():
            raise HTTPException(status_code=422, detail=f"file not found: {file_path}")
        return file_path

    @property
    def thumbnail_directory(self) -> Path:
        return self.output_dir / "thumbnails"

    @staticmethod
    def _media_names(directory: Path) -> List[MediasResponse]:
        if directory is None:
            return []
        names = sorted([it.name for it in glob_img(directory)])
        res = []
        for name in names:
            path = os.path.join(directory, name)
            img = Image.open(path)
            res.append(
                MediasResponse(
                    name=name,
                    height=img.height,
                    width=img.width,
                    ctime=os.path.getctime(path),
                    mtime=os.path.getmtime(path),
                )
            )
        return res

    def get_thumbnail(
        self, directory: Path, original_filename: str, width, height, **options
    ):
        directory = Path(directory)
        storage = FilesystemStorageBackend(self.app)
        crop = options.get("crop", "fit")
        background = options.get("background")
        quality = options.get("quality", 90)

        original_path, original_filename = os.path.split(original_filename)
        original_filepath = os.path.join(directory, original_path, original_filename)
        image = Image.open(BytesIO(storage.read(original_filepath)))

        # keep ratio resize
        if not width and not height:
            width = 256

        if width != 0:
            height = int(image.height * width / image.width)
        else:
            width = int(image.width * height / image.height)

        thumbnail_size = (width, height)

        thumbnail_filename = generate_filename(
            directory,
            original_filename,
            aspect_to_string(thumbnail_size),
            crop,
            background,
            quality,
        )

        thumbnail_filepath = os.path.join(
            self.thumbnail_directory, original_path, thumbnail_filename
        )

        if storage.exists(thumbnail_filepath):
            return thumbnail_filepath, (width, height)

        try:
            image.load()
        except (IOError, OSError):
            self.app.logger.warning("Thumbnail not load image: %s", original_filepath)
            return thumbnail_filepath, (width, height)

        # get original image format
        options["format"] = options.get("format", image.format)

        image = self._create_thumbnail(
            image, thumbnail_size, crop, background=background
        )

        raw_data = self.get_raw_data(image, **options)
        storage.save(thumbnail_filepath, raw_data)

        return thumbnail_filepath, (width, height)

    def get_raw_data(self, image, **options):
        data = {
            "format": self._get_format(image, **options),
            "quality": options.get("quality", 90),
        }

        _file = BytesIO()
        image.save(_file, **data)
        return _file.getvalue()

    @staticmethod
    def colormode(image, colormode="RGB"):
        if colormode == "RGB" or colormode == "RGBA":
            if image.mode == "RGBA":
                return image
            if image.mode == "LA":
                return image.convert("RGBA")
            return image.convert(colormode)

        if colormode == "GRAY":
            return image.convert("L")

        return image.convert(colormode)

    @staticmethod
    def background(original_image, color=0xFF):
        size = (max(original_image.size),) * 2
        image = Image.new("L", size, color)
        image.paste(
            original_image,
            tuple(map(lambda x: (x[0] - x[1]) / 2, zip(size, original_image.size))),
        )

        return image

    def _get_format(self, image, **options):
        if options.get("format"):
            return options.get("format")
        if image.format:
            return image.format

        return "JPEG"

    def _create_thumbnail(self, image, size, crop="fit", background=None):
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:  # pylint: disable=raise-missing-from
            resample = Image.ANTIALIAS

        if crop == "fit":
            image = ImageOps.fit(image, size, resample)
        else:
            image = image.copy()
            image.thumbnail(size, resample=resample)

        if background is not None:
            image = self.background(image)

        image = self.colormode(image)

        return image
