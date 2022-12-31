# Copy from https://github.com/silentsokolov/flask-thumbnails/blob/master/flask_thumbnails/thumbnail.py
import os
from functools import lru_cache
from io import BytesIO

from PIL import Image, ImageOps, PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
from .storage_backends import FilesystemStorageBackend
from .utils import aspect_to_string, generate_filename, glob_img


class FileManager:
    def __init__(self, app=None):
        self.app = app
        self._default_root_directory = "media"
        self._default_thumbnail_directory = "media"
        self._default_root_url = "/"
        self._default_thumbnail_root_url = "/"
        self._default_format = "JPEG"

        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        if self.app is None:
            self.app = app
        app.thumbnail_instance = self

        if not hasattr(app, "extensions"):
            app.extensions = {}

        if "thumbnail" in app.extensions:
            raise RuntimeError("Flask-thumbnail extension already initialized")

        app.extensions["thumbnail"] = self

        app.config.setdefault("THUMBNAIL_MEDIA_ROOT", self._default_root_directory)
        app.config.setdefault("THUMBNAIL_MEDIA_THUMBNAIL_ROOT", self._default_thumbnail_directory)
        app.config.setdefault("THUMBNAIL_MEDIA_URL", self._default_root_url)
        app.config.setdefault("THUMBNAIL_MEDIA_THUMBNAIL_URL", self._default_thumbnail_root_url)
        app.config.setdefault("THUMBNAIL_DEFAULT_FORMAT", self._default_format)

    @property
    def root_directory(self):
        path = self.app.config["THUMBNAIL_MEDIA_ROOT"]

        if os.path.isabs(path):
            return path
        else:
            return os.path.join(self.app.root_path, path)

    @property
    def thumbnail_directory(self):
        path = self.app.config["THUMBNAIL_MEDIA_THUMBNAIL_ROOT"]

        if os.path.isabs(path):
            return path
        else:
            return os.path.join(self.app.root_path, path)

    @property
    def root_url(self):
        return self.app.config["THUMBNAIL_MEDIA_URL"]

    @property
    @lru_cache()
    def media_names(self):
        names = sorted([it.name for it in glob_img(self.root_directory)])
        res = []
        for name in names:
            img = Image.open(os.path.join(self.root_directory, name))
            res.append({"name": name, "height": img.height, "width": img.width})
        return res

    @property
    def thumbnail_url(self):
        return self.app.config["THUMBNAIL_MEDIA_THUMBNAIL_URL"]

    def get_thumbnail(self, original_filename, width, height, **options):
        storage = FilesystemStorageBackend(self.app)
        crop = options.get("crop", "fit")
        background = options.get("background")
        quality = options.get("quality", 90)

        original_path, original_filename = os.path.split(original_filename)
        original_filepath = os.path.join(self.root_directory, original_path, original_filename)
        image = Image.open(BytesIO(storage.read(original_filepath)))

        # keep ratio resize
        if width is not None:
            height = int(image.height * width / image.width)
        else:
            width = int(image.width * height / image.height)

        thumbnail_size = (width, height)

        thumbnail_filename = generate_filename(
            original_filename, aspect_to_string(thumbnail_size), crop, background, quality
        )

        thumbnail_filepath = os.path.join(
            self.thumbnail_directory, original_path, thumbnail_filename
        )
        thumbnail_url = os.path.join(self.thumbnail_url, original_path, thumbnail_filename)

        if storage.exists(thumbnail_filepath):
            return thumbnail_url, (width, height)

        try:
            image.load()
        except (IOError, OSError):
            self.app.logger.warning("Thumbnail not load image: %s", original_filepath)
            return thumbnail_url, (width, height)

        # get original image format
        options["format"] = options.get("format", image.format)

        image = self._create_thumbnail(image, thumbnail_size, crop, background=background)

        raw_data = self.get_raw_data(image, **options)
        storage.save(thumbnail_filepath, raw_data)

        return thumbnail_url, (width, height)

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

        return self.app.config["THUMBNAIL_DEFAULT_FORMAT"]

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
