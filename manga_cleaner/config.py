"""
Centralized configuration for Manga Cleaner.

All detection thresholds and runtime settings live here as attributes of a
single ``Config`` instance.  Every module imports the shared ``config``
singleton instead of relying on module-level globals, and the GUI updates
the same instance before launching a processing run.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    """Mutable configuration container — one instance is shared app-wide."""

    # -- Paths ---------------------------------------------------------------
    root_folder: str = r"C:\path\to\your\manga"

    # -- Supported image formats ---------------------------------------------
    supported_extensions: set[str] = field(
        default_factory=lambda: {
            ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif",
        }
    )

    # -- Blank-page detection (0–255 grayscale) ------------------------------
    white_threshold: int = 250
    black_threshold: int = 5
    solid_ratio: float = 0.99

    # -- Duplicate detection -------------------------------------------------
    hash_threshold: int = 8

    # -- Dry-run mode --------------------------------------------------------
    dry_run: bool = True

    # -- Aspect ratio filter -------------------------------------------------
    aspect_ratio_min: float = 0.55
    aspect_ratio_max: float = 0.80

    # -- Saturation filter ---------------------------------------------------
    saturation_threshold: float = 0.18

    # -- Size outlier filter -------------------------------------------------
    size_outlier_z: float = 2.5

    # -- Text-only page filter -----------------------------------------------
    text_bg_min: int = 230
    text_std_max: int = 40
    text_edge_ratio_max: float = 0.03

    # -- ML (CLIP) classifier ------------------------------------------------
    ml_confidence_threshold: float = 0.75
    ml_manga_label: str = (
        "a black and white Japanese manga comic book page with sequential "
        "art panels, speech bubbles, and narrative storytelling"
    )
    ml_fanart_label: str = (
        "a colorful anime fan art, vibrant illustration, promotional artwork, "
        "or advertisement with bright colors and no speech bubbles"
    )

    # -- Folder discovery ----------------------------------------------------
    max_scan_depth: int = 4


# Module-level singleton — import this everywhere.
config = Config()

