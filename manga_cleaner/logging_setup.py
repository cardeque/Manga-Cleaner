"""
Logging setup for Manga Cleaner.

Provides a module-level ``log`` logger and a ``setup_logging()`` function
that configures console + file output.  The setup is deferred so that
merely importing the package does not create ``cleaning_log.txt``.
"""

import logging
import os

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cleaning_log.txt")

log = logging.getLogger("manga_cleaner")

_logging_configured = False


def setup_logging() -> None:
    """Configure the ``manga_cleaner`` logger (console + file).

    Safe to call multiple times — only the first call has any effect.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    log.addHandler(console)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)


class TextHandler(logging.Handler):
    """Tiny logging handler that forwards records to a callable."""

    def __init__(self, callback):
        super().__init__()
        self._cb = callback

    def emit(self, record):
        self._cb(self.format(record))

