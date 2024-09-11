import logging

LOGGING_FORMAT = f'%(name)s:     %(asctime)s - %(message)s'

logging.basicConfig(
    level=logging.WARNING,
    format=LOGGING_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("APP LOG")