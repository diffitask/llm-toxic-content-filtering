import logging

LOGGING_FORMAT = f'%(asctime)s - %(message)s'

logging.basicConfig(
    filename='alert.log',
    filemode='a',
    level=logging.WARNING,
    format=LOGGING_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger("APP LOG")