import logging
import logging.config


def configure_logging():
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger
