"""
Advanced detection helpers for Manga Cleaner.

Heuristic filters that flag non-manga content based on aspect ratio, colour
saturation, and resolution outliers.  These do **not** use ML models — see
:mod:`manga_cleaner.ml_classifier` for CLIP-based detection.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageStat

from manga_cleaner.config import config
from manga_cleaner.image_helpers import load_image
from manga_cleaner.logging_setup import log


# ---------------------------------------------------------------------------
# Aspect ratio
# ---------------------------------------------------------------------------

def is_wrong_aspect_ratio(img: Image.Image) -> bool:
    """Flag images whose width/height ratio falls outside the expected
    manga portrait range (``config.aspect_ratio_min`` – ``config.aspect_ratio_max``).
    """
    if img.height == 0:
        return False
    ratio = img.width / img.height
    return not (config.aspect_ratio_min <= ratio <= config.aspect_ratio_max)


# ---------------------------------------------------------------------------
# Colour saturation
# ---------------------------------------------------------------------------

def is_high_saturation(img: Image.Image) -> bool:
    """Flag images with unusually high average colour saturation.

    Manga pages are mostly B&W (low saturation); colourful fan-art is not.
    """
    try:
        import numpy as np
        arr = np.array(img.convert("HSV"))
        avg_sat = arr[:, :, 1].mean() / 255.0
        return avg_sat > config.saturation_threshold
    except ImportError:
        # numpy not available — fall back to a Pillow-only approach
        r, g, b = img.split()
        stat_r = ImageStat.Stat(r).mean[0]
        stat_g = ImageStat.Stat(g).mean[0]
        stat_b = ImageStat.Stat(b).mean[0]
        max_c = max(stat_r, stat_g, stat_b)
        min_c = min(stat_r, stat_g, stat_b)
        sat = (max_c - min_c) / max_c if max_c > 0 else 0.0
        return (sat / 255.0) > config.saturation_threshold


# ---------------------------------------------------------------------------
# Size outlier
# ---------------------------------------------------------------------------

def find_size_outliers(image_paths: list[Path]) -> dict[Path, str]:
    """Flag images whose pixel area is a statistical outlier (z-score >
    ``config.size_outlier_z``) within the chapter.

    Requires at least 4 images to compute meaningful statistics.
    """
    try:
        import numpy as np
    except ImportError:
        return {}

    if len(image_paths) < 4:
        return {}

    def _get_size(p: Path):
        img = load_image(p)
        if img:
            return p, img.width * img.height
        return p, None

    sizes: list[tuple[Path, int]] = []
    max_workers = min(os.cpu_count() or 4, len(image_paths), 8)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for p, area in pool.map(_get_size, image_paths):
            if area is not None:
                sizes.append((p, area))

    if len(sizes) < 4:
        return {}

    areas = np.array([s for _, s in sizes], dtype=float)
    mean, std = areas.mean(), areas.std()
    if std == 0:
        return {}

    return {
        p: f"size outlier (area={area:.0f}, z={(area - mean) / std:+.1f})"
        for p, area in sizes
        if abs((area - mean) / std) > config.size_outlier_z
    }

