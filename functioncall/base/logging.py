import logging
from typing import Optional

import colorlog

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
LOGLEVEL = logging.DEBUG

formatter = colorlog.ColoredFormatter(
    fmt="%(log_color)s" + LOG_FORMAT,
    datefmt=DATE_FORMAT,
    log_colors={
        "DEBUG": "blue",
        "INFO": "light_purple",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_white,bg_red",
    },
)

handler = logging.StreamHandler()
handler.setLevel(LOGLEVEL)
handler.setFormatter(formatter)

logging.basicConfig(level=LOGLEVEL, handlers=[handler])


def getLogger(name: Optional[str] = None):
    return logging.getLogger(name)
