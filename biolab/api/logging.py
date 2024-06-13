import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing
import datasets

# Setup datasets logging level, feels weird to do here, but neccesary
datasets.logging.set_verbosity_error()
datasets.disable_progress_bars()

# Defaults to None, inside the worker function call `setup_logging` to init
logging_queue = None


def setup_logging(use_queue: bool = False) -> logging.Logger:
    """Set up logging for the evaluation workflow."""
    logger = logging.getLogger("biolab")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if use_queue:
        global logging_queue
        logging_queue = multiprocessing.Queue()
        queue_handler = QueueHandler(logging_queue)
        queue_handler.setFormatter(formatter)
        queue_handler.setLevel(logging.DEBUG)
        logger.addHandler(queue_handler)

        listener = QueueListener(logging_queue, logging.StreamHandler())
        listener.start()
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


logger = setup_logging()
