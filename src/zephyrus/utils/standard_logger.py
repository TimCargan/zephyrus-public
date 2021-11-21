# Make logger
import logging
import sys
from abc import ABC


def build_logger(name=None):
    name = __name__ if name is None else name
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d: %(levelname)s %(name)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)

    logger.handlers = [ch]
    if not logger.hasHandlers():
        logger.addHandler(ch)
    return logger


logger = build_logger()


class Logged(ABC):
    def __init__(self):
        self.logger = build_logger()

