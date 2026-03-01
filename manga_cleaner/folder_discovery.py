"""
Recursive folder discovery for Manga Cleaner.

Walks a directory tree (up to a configurable depth) and returns every folder
that directly contains at least one supported image file.
"""

from pathlib import Path

from manga_cleaner.config import config
from manga_cleaner.logging_setup import log


def find_image_folders(root: Path, max_depth: int | None = None) -> list[Path]:
    """Recursively find folders that contain images.

    Parameters
    ----------
    root : Path
        Top-level directory to start from.
    max_depth : int, optional
        Override ``config.max_scan_depth``.

    Returns
    -------
    list[Path]
        Sorted list of directories containing at least one image file.
    """
    if max_depth is None:
        max_depth = config.max_scan_depth

    result: list[Path] = []

    def _walk(folder: Path, depth: int):
        if depth > max_depth:
            return
        has_images = any(
            f.is_file() and f.suffix.lower() in config.supported_extensions
            for f in folder.iterdir()
        )
        if has_images:
            result.append(folder)
        try:
            for child in sorted(folder.iterdir()):
                if child.is_dir():
                    _walk(child, depth + 1)
        except PermissionError:
            log.warning(f"  [WARN] Permission denied: {folder}")

    _walk(root, 0)
    return result

