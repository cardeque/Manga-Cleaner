"""
Manga Cleaner — detect and remove unwanted images from manga folders.

Run with::

    python -m manga_cleaner

Or import and call :func:`main` directly.
"""

__version__ = "1.0.0"


def main() -> None:
    """Launch the Manga Cleaner GUI."""
    from manga_cleaner.logging_setup import setup_logging
    setup_logging()

    from manga_cleaner.gui import CleanerApp
    app = CleanerApp()
    app.mainloop()

