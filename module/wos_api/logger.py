import logging

logging.basicConfig()
logger = logging.getLogger("wos-sdk")


def enable_log(level=logging.DEBUG):
    """Enable log output for WOS SDK"""
    logger.setLevel(level)
