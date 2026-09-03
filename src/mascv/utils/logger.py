"""Application logging setup."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Retrieve configured logger instance."""
    return logging.getLogger(name)
