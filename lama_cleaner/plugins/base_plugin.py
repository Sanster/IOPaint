from loguru import logger
import numpy as np

from lama_cleaner.schema import RunPluginRequest


class BasePlugin:
    def __init__(self):
        err_msg = self.check_dep()
        if err_msg:
            logger.error(err_msg)
            exit(-1)

    def __call__(self, rgb_np_img, req: RunPluginRequest) -> np.array:
        # return RGBA np image or BGR np image
        ...

    def check_dep(self):
        ...
