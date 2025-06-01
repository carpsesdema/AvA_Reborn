import logging

LOG_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"

def init_logging(level=logging.DEBUG):
    logging.basicConfig(format=LOG_FORMAT, level=level)

def get_logger(name: str):
    return logging.getLogger(name)