"""
Low-level image utilities for Manga Cleaner.

Functions for loading images, detecting blank pages, computing hashes, and
finding duplicates within a list of image paths.
"""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError

from manga_cleaner.config import config
from manga_cleaner.logging_setup import log


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image(path: Path) -> Image.Image | None:
    """Open an image and convert to RGB.  Returns ``None`` on failure."""
    try:
        img = Image.open(path)
        img.load()  # force full decode so corrupted files fail here
        return img.convert("RGB")
    except UnidentifiedImageError:
        log.warning(f"  [WARN] Cannot identify image (skipped): {path.name}")
    except Exception as exc:
        log.warning(f"  [WARN] Could not open {path.name}: {exc}")
    return None


# ---------------------------------------------------------------------------
# Blank-page detection
# ---------------------------------------------------------------------------

def is_blank_page(img: Image.Image) -> str | None:
    """Detect fully-white or fully-black pages.

    Returns ``'white'``, ``'black'``, or ``None``.
    Uses Pillow's histogram to count pixels above/below the configured
    thresholds without loading every pixel into a Python loop.
    """
    gray = img.convert("L")
    total_pixels = gray.width * gray.height
    if total_pixels == 0:
        return None

    hist = gray.histogram()

    white_pixels = sum(hist[config.white_threshold:])
    black_pixels = sum(hist[: config.black_threshold + 1])

    if white_pixels / total_pixels >= config.solid_ratio:
        return "white"
    if black_pixels / total_pixels >= config.solid_ratio:
        return "black"
    return None


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_hashes(img: Image.Image) -> tuple[str, "imagehash.ImageHash"]:
    """Return ``(exact_hash, perceptual_hash)`` for an image.

    *exact_hash* — MD5 of raw RGB pixel bytes (byte-for-byte duplicates).
    *perceptual* — pHash (visually identical images at different resolutions).
    """
    exact = hashlib.md5(img.tobytes()).hexdigest()
    perceptual = imagehash.phash(img)
    return exact, perceptual


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def find_duplicates(image_paths: list[Path]) -> dict[Path, str]:
    """Identify duplicate images within *image_paths*.

    Strategy
    --------
    1. Exact duplicates (same MD5 of pixel data) are caught first.
    2. Near-duplicates (perceptual hash distance ≤ threshold) are detected
       among the survivors.

    Within each duplicate group the image with the *lowest sorted filename*
    is kept; all others are flagged for removal.

    Returns a dict mapping each duplicate ``Path`` → human-readable reason.
    """
    duplicates: dict[Path, str] = {}

    loaded: list[tuple[Path, str, "imagehash.ImageHash"]] = []

    def _load_and_hash(p: Path):
        img = load_image(p)
        if img is None:
            return p, None, None
        exact, phash = compute_hashes(img)
        return p, exact, phash

    max_workers = min(os.cpu_count() or 4, len(image_paths), 8)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {pool.submit(_load_and_hash, p): p for p in image_paths}
        for future in as_completed(futures):
            p, exact, phash = future.result()
            if exact is None:
                duplicates[p] = "corrupted"
            else:
                loaded.append((p, exact, phash))

    # --- Pass 1: exact duplicates (same MD5) ---
    exact_groups: dict[str, list[Path]] = {}
    for p, exact, _ in loaded:
        exact_groups.setdefault(exact, []).append(p)

    survivor_paths: set[Path] = set()
    for group in exact_groups.values():
        group_sorted = sorted(group, key=lambda x: x.name)
        survivor_paths.add(group_sorted[0])
        for dup in group_sorted[1:]:
            duplicates[dup] = f"exact duplicate of '{group_sorted[0].name}'"

    survivors = [(p, e, ph) for p, e, ph in loaded if p in survivor_paths]

    # --- Pass 2: near-duplicates (perceptual hash) ---
    used: set[Path] = set()
    for i in range(len(survivors)):
        p1, _, ph1 = survivors[i]
        if p1 in used:
            continue
        for j in range(i + 1, len(survivors)):
            p2, _, ph2 = survivors[j]
            if p2 in used:
                continue
            dist = ph1 - ph2
            if dist <= config.hash_threshold:
                keeper, dup = sorted([p1, p2], key=lambda x: x.name)
                duplicates[dup] = f"near-duplicate of '{keeper.name}' (hash dist={dist})"
                used.add(dup)

    return duplicates

