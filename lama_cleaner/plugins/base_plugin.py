from loguru import logger


class BasePlugin:
    def __init__(self):
        err_msg = self.check_dep()
        if err_msg:
            logger.error(err_msg)
            exit(-1)

    def __call__(self, rgb_np_img, files, form):
        ...

    def check_dep(self):
        ...
