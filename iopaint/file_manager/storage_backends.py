# Copy from https://github.com/silentsokolov/flask-thumbnails/blob/master/flask_thumbnails/storage_backends.py
import errno
import os
from abc import ABC, abstractmethod


class BaseStorageBackend(ABC):
    def __init__(self, app=None):
        self.app = app

    @abstractmethod
    def read(self, filepath, mode="rb", **kwargs):
        raise NotImplementedError

    @abstractmethod
    def exists(self, filepath):
        raise NotImplementedError

    @abstractmethod
    def save(self, filepath, data):
        raise NotImplementedError


class FilesystemStorageBackend(BaseStorageBackend):
    def read(self, filepath, mode="rb", **kwargs):
        with open(filepath, mode) as f:  # pylint: disable=unspecified-encoding
            return f.read()

    def exists(self, filepath):
        return os.path.exists(filepath)

    def save(self, filepath, data):
        directory = os.path.dirname(filepath)

        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        if not os.path.isdir(directory):
            raise IOError("{} is not a directory".format(directory))

        with open(filepath, "wb") as f:
            f.write(data)
