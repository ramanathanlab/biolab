from __future__ import annotations  # noqa: D100

import logging

import datasets

# Setup datasets logging level
datasets.logging.set_verbosity_error()
datasets.disable_progress_bars()


def setup_logging() -> logging.Logger:
    """Set up logging for the evaluation workflow."""
    logger = logging.getLogger('biolab')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


# Initialize logger once when the module is imported
logger = setup_logging()
