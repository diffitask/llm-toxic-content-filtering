import logging

# configure logging: saving 1) initial user prompt, 2) filtering model answer
LOGGING_FORMAT = f'%(name)s:     %(asctime)s - %(message)s'

logging.basicConfig(
    level=logging.WARNING,
    format=LOGGING_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("APP LOG")