import os
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageTk, UnidentifiedImageError, ImageStat
import imagehash
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Root folder containing one subfolder per chapter/volume
ROOT_FOLDER = r"C:\path\to\your\manga"

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

# Blank-page detection thresholds (0-255 grayscale)
WHITE_THRESHOLD = 250  # pixels at or above this are considered "white"
BLACK_THRESHOLD = 5  # pixels at or below this are considered "black"
SOLID_RATIO = 0.99  # fraction of pixels that must be solid to flag the image

# Duplicate detection — maximum perceptual hash distance to consider two images duplicates
# 0 = identical hash, higher = more tolerant. 8–10 is a good balance for manga.
HASH_THRESHOLD = 8

# Set to True to only preview what would be deleted without touching any files
DRY_RUN = True

# ---------------------------------------------------------------------------
# ADVANCED DETECTION — defaults (overridden by GUI)
# ---------------------------------------------------------------------------

# --- Aspect ratio filter ---
# Typical manga portrait page: width/height roughly 0.55–0.80
ASPECT_RATIO_MIN = 0.55
ASPECT_RATIO_MAX = 0.80

# --- Color saturation filter ---
# Manga is mostly B&W; fan-art tends to be highly colorful.
# Average HSV-saturation (0.0–1.0) above this threshold flags the image.
SATURATION_THRESHOLD = 0.18

# --- Size outlier filter ---
# Pages whose pixel-area z-score exceeds this are flagged as resolution outliers.
SIZE_OUTLIER_Z = 2.5

# --- Text-only page filter ---
# A page is considered "text-only" when:
#   • its average brightness is very high (mostly white background)      → TEXT_BG_MIN
#   • AND its std-dev of brightness is low (little dark ink variation)   → TEXT_STD_MAX
#   • AND it has very few dark "edge" pixels (no panel borders / art)    → TEXT_EDGE_RATIO_MAX
TEXT_BG_MIN = 230  # avg grayscale brightness must be >= this
TEXT_STD_MAX = 40  # std-dev of grayscale must be <= this
TEXT_EDGE_RATIO_MAX = 0.03  # fraction of edge pixels must be <= this

# --- ML (CLIP) classifier ---
# Confidence that the image is NOT a manga page needed to flag it.
ML_CONFIDENCE_THRESHOLD = 0.75
# CLIP labels used for zero-shot classification — detailed descriptions work best
ML_MANGA_LABEL = "a black and white Japanese manga comic book page with sequential art panels, speech bubbles, and narrative storytelling"
ML_FANART_LABEL = "a colorful anime fan art, vibrant illustration, promotional artwork, or advertisement with bright colors and no speech bubbles"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

LOG_FILE = os.path.join(os.path.dirname(__file__), "cleaning_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IMAGE HELPERS
# ---------------------------------------------------------------------------

def load_image(path: Path) -> Image.Image | None:
    """Open an image and convert to RGB. Returns None if the file is corrupted."""
    try:
        img = Image.open(path)
        img.load()  # force full decode so corrupted files fail here
        return img.convert("RGB")
    except UnidentifiedImageError:
        log.warning(f"  [WARN] Cannot identify image (skipped): {path.name}")
    except Exception as exc:
        log.warning(f"  [WARN] Could not open {path.name}: {exc}")
    return None


def is_blank_page(img: Image.Image) -> str | None:
    """
    Detect fully-white or fully-black pages.

    Returns 'white', 'black', or None.
    Uses Pillow's histogram to count pixels above/below the thresholds
    without loading every pixel into a Python loop.
    """
    gray = img.convert("L")
    total_pixels = gray.width * gray.height
    if total_pixels == 0:
        return None

    # hist is a list of 256 counts, one per brightness level (0=black, 255=white)
    hist = gray.histogram()

    white_pixels = sum(hist[WHITE_THRESHOLD:])  # brightness >= WHITE_THRESHOLD
    black_pixels = sum(hist[: BLACK_THRESHOLD + 1])  # brightness <= BLACK_THRESHOLD

    if white_pixels / total_pixels >= SOLID_RATIO:
        return "white"
    if black_pixels / total_pixels >= SOLID_RATIO:
        return "black"
    return None


def compute_hashes(img: Image.Image) -> tuple[str, "imagehash.ImageHash"]:
    """
    Returns (exact_hash, perceptual_hash) for an image.

    exact_hash  — MD5 of raw RGB pixel bytes; catches byte-for-byte identical images.
    perceptual  — pHash; catches visually identical images saved at different
                  resolutions or compression levels.
    """
    exact = hashlib.md5(img.tobytes()).hexdigest()
    perceptual = imagehash.phash(img)
    return exact, perceptual


def find_duplicates(image_paths: list[Path]) -> dict[Path, str]:
    """
    Identify duplicate images within a list of paths.

    Strategy:
      1. Exact duplicates (same MD5 of pixel data) are caught first.
      2. Near-duplicates (perceptual hash distance <= HASH_THRESHOLD) are
         detected among the survivors.

    Within each duplicate group the image with the *lowest sorted filename*
    is kept; all others are flagged for removal.

    Returns a dict mapping each duplicate path -> human-readable reason string.
    """
    duplicates: dict[Path, str] = {}

    # Load all images and compute both hashes in parallel
    loaded: list[tuple[Path, str, "imagehash.ImageHash"]] = []  # (path, exact_md5, phash)

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
        survivor_paths.add(group_sorted[0])  # keep the first alphabetically
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
            if dist <= HASH_THRESHOLD:
                keeper, dup = sorted([p1, p2], key=lambda x: x.name)
                duplicates[dup] = f"near-duplicate of '{keeper.name}' (hash dist={dist})"
                used.add(dup)

    return duplicates


# ---------------------------------------------------------------------------
# ADVANCED DETECTION HELPERS
# ---------------------------------------------------------------------------

# Lock for thread-safe access to the CLIP model during parallel inference
_clip_lock = threading.Lock()


def is_wrong_aspect_ratio(img: Image.Image) -> bool:
    """
    Flag images whose width/height ratio falls outside the expected manga
    portrait range (ASPECT_RATIO_MIN – ASPECT_RATIO_MAX).
    Landscape or square pages (fan-art, ads) are caught here.
    """
    if img.height == 0:
        return False
    ratio = img.width / img.height
    return not (ASPECT_RATIO_MIN <= ratio <= ASPECT_RATIO_MAX)


def is_high_saturation(img: Image.Image) -> bool:
    """
    Flag images with unusually high average color saturation.
    Manga pages are mostly B&W (low saturation); colorful fan-art is not.
    Uses Pillow's ImageStat on the S-channel of the HSV image.
    """
    try:
        import numpy as np
        arr = np.array(img.convert("HSV"))
        avg_sat = arr[:, :, 1].mean() / 255.0
        return avg_sat > SATURATION_THRESHOLD
    except ImportError:
        # numpy not available — fall back to a Pillow-only approach
        r, g, b = img.split()
        stat_r = ImageStat.Stat(r).mean[0]
        stat_g = ImageStat.Stat(g).mean[0]
        stat_b = ImageStat.Stat(b).mean[0]
        max_c = max(stat_r, stat_g, stat_b)
        min_c = min(stat_r, stat_g, stat_b)
        sat = (max_c - min_c) / max_c if max_c > 0 else 0.0
        return (sat / 255.0) > SATURATION_THRESHOLD


def find_size_outliers(image_paths: list[Path]) -> dict[Path, str]:
    """
    Flag images whose pixel area is a statistical outlier (z-score >
    SIZE_OUTLIER_Z) within the chapter.  Pages scanned at a completely
    different resolution than their siblings are likely foreign inserts.
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
        if abs((area - mean) / std) > SIZE_OUTLIER_Z
    }


def is_text_only_page(img: Image.Image) -> tuple[bool, float]:
    """
    Detect pages that contain only text and no artwork using CLIP ML model.

    Uses CLIP zero-shot classification to detect text-only pages
    (survey, table of contents, afterword, credits).

    Returns (is_text_only: bool, confidence: float).
    Raises no exceptions — failures are logged and return (False, 0.0).
    """
    try:
        import torch
        with _clip_lock:
            model, processor = _get_clip_model()

            # Labels for text-only vs manga page detection
            text_label = "a mostly white page with only Japanese or English text, such as table of contents, survey form, author notes, afterword, or credits page with no artwork"
            manga_label = "a black and white Japanese manga comic book page with sequential art panels, character illustrations, speech bubbles, and visual storytelling"

            inputs = processor(
                text=[manga_label, text_label],
                images=img,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]

        text_prob = probs[1].item()
        # Use the same thresholds as the ML non-manga detection
        return text_prob >= ML_CONFIDENCE_THRESHOLD, text_prob

    except ImportError:
        log.warning("  [WARN] torch/transformers not installed — text-only ML check skipped.")
        return False, 0.0
    except Exception as exc:
        log.warning(f"  [WARN] CLIP text-only classification error: {exc}")
        return False, 0.0


# ---------------------------------------------------------------------------
# ML (CLIP) CLASSIFIER
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_clip_model():
    """
    Load the CLIP model and processor exactly once, then cache them.
    The first call takes ~10–30 s (download + load); subsequent calls are instant.
    """
    from transformers import CLIPProcessor, CLIPModel
    model_name = "openai/clip-vit-base-patch32"
    log.info("  [ML] Loading CLIP model (first run may take a moment)…")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    log.info("  [ML] CLIP model ready.")
    return model, processor


def is_non_manga_ml(img: Image.Image) -> tuple[bool, float]:
    """
    Use CLIP zero-shot classification to detect non-manga content
    (fan-art, advertisements, colorful illustrations).

    Returns (is_foreign: bool, fanart_confidence: float).
    Raises no exceptions — failures are logged and return (False, 0.0).
    """
    try:
        import torch
        with _clip_lock:
            model, processor = _get_clip_model()

            inputs = processor(
                text=[ML_MANGA_LABEL, ML_FANART_LABEL],
                images=img,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]

        fanart_prob = probs[1].item()
        return fanart_prob >= ML_CONFIDENCE_THRESHOLD, fanart_prob

    except ImportError:
        log.warning("  [WARN] torch/transformers not installed — ML check skipped.")
        return False, 0.0
    except Exception as exc:
        log.warning(f"  [WARN] CLIP classification error: {exc}")
        return False, 0.0


# ---------------------------------------------------------------------------
# FOLDER DISCOVERY
# ---------------------------------------------------------------------------

MAX_SCAN_DEPTH = 4  # maximum folder depth to recurse into


def find_image_folders(root: Path, max_depth: int = MAX_SCAN_DEPTH) -> list[Path]:
    """
    Recursively find all folders (up to *max_depth* levels below *root*)
    that directly contain at least one supported image file.

    The root itself is included if it contains images.
    Results are sorted alphabetically by full path.
    """
    result: list[Path] = []

    def _walk(folder: Path, depth: int):
        if depth > max_depth:
            return
        # Check if this folder itself has images
        has_images = any(
            f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            for f in folder.iterdir()
        )
        if has_images:
            result.append(folder)
        # Recurse into subfolders
        try:
            for child in sorted(folder.iterdir()):
                if child.is_dir():
                    _walk(child, depth + 1)
        except PermissionError:
            log.warning(f"  [WARN] Permission denied: {folder}")

    _walk(root, 0)
    return result


# ---------------------------------------------------------------------------
# CHAPTER PROCESSOR
# ---------------------------------------------------------------------------

def process_chapter(
        chapter_path: Path,
        dry_run: bool,
        use_aspect: bool = False,
        use_saturation: bool = False,
        use_size_outlier: bool = False,
        use_text_only: bool = False,
        use_ml: bool = False,
        collect_for_preview: bool = False,
        confirm_callback=None,
        stop_event: threading.Event | None = None,
) -> dict:
    """
    Scan one chapter folder, detect blank/duplicate/foreign/text-only images,
    and (unless dry_run) delete them after user confirmation.

    collect_for_preview — if True, skip deletion and return the flagged files
    in stats["pending_delete"] and all image paths in stats["all_images"]
    so the caller can batch-preview across multiple chapters.

    confirm_callback — if provided, called with (chapter_name, file_list_text,
    count) and must return True/False.  Used to show the confirmation dialog
    on the main thread when processing runs in a background thread.

    stop_event — if provided and set, the function returns early with partial
    results.

    Returns a stats dict: scanned, blank, duplicate, aspect, saturation,
    size_outlier, text_only, ml_foreign, removed, skipped,
    and optionally pending_delete / all_images.
    """
    log.info(f"  🔍 Analyzing folder: {chapter_path.name}")

    stats = {
        "scanned": 0, "blank": 0, "duplicate": 0,
        "aspect": 0, "saturation": 0, "size_outlier": 0,
        "text_only": 0, "ml_foreign": 0,
        "removed": 0, "skipped": 0,
    }

    image_files = sorted(
        p for p in chapter_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        log.info("  (no images found)\n")
        return stats

    stats["scanned"] = len(image_files)

    # Helper to check if stop was requested
    def _stopped():
        return stop_event is not None and stop_event.is_set()

    to_delete: dict[Path, str] = {}  # path -> reason
    non_blank: list[Path] = []

    # --- Detect blank pages (parallel) ---
    def _check_blank(p: Path):
        img = load_image(p)
        if img is None:
            return p, "corrupted"
        blank = is_blank_page(img)
        if blank:
            return p, f"blank ({blank})"
        return p, None

    max_workers = min(os.cpu_count() or 4, len(image_files), 8)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {pool.submit(_check_blank, p): p for p in image_files}
        for future in as_completed(futures):
            p, reason = future.result()
            if reason:
                to_delete[p] = reason
                if reason != "corrupted":
                    stats["blank"] += 1
            else:
                non_blank.append(p)
    non_blank.sort(key=lambda x: x.name)  # restore sorted order

    if _stopped():
        return stats

    # --- Detect duplicates among non-blank survivors ---
    for p, reason in find_duplicates(non_blank).items():
        if reason == "corrupted":
            to_delete[p] = "corrupted"
        else:
            to_delete[p] = reason
            stats["duplicate"] += 1

    if _stopped():
        return stats

    # Build a working set of survivors for the advanced passes
    survivors = [p for p in non_blank if p not in to_delete]

    # --- Aspect ratio filter (parallel) ---
    if use_aspect and not _stopped():
        def _check_aspect(p: Path):
            img = load_image(p)
            if img and is_wrong_aspect_ratio(img):
                return p, f"wrong aspect ratio ({img.width}×{img.height}, ratio={img.width / img.height:.2f})"
            return p, None

        with ThreadPoolExecutor(max_workers=max(1, min(os.cpu_count() or 4, len(survivors), 8))) as pool:
            futures = {pool.submit(_check_aspect, p): p for p in survivors}
            flagged_aspect = []
            for future in as_completed(futures):
                p, reason = future.result()
                if reason:
                    flagged_aspect.append((p, reason))
        for p, reason in flagged_aspect:
            to_delete[p] = reason
            stats["aspect"] += 1
            survivors.remove(p)

    # --- Color saturation filter (parallel) ---
    if use_saturation and not _stopped():
        def _check_saturation(p: Path):
            img = load_image(p)
            if img and is_high_saturation(img):
                return p, "high color saturation (likely fan-art / color insert)"
            return p, None

        with ThreadPoolExecutor(max_workers=max(1, min(os.cpu_count() or 4, len(survivors), 8))) as pool:
            futures = {pool.submit(_check_saturation, p): p for p in survivors}
            flagged_sat = []
            for future in as_completed(futures):
                p, reason = future.result()
                if reason:
                    flagged_sat.append((p, reason))
        for p, reason in flagged_sat:
            to_delete[p] = reason
            stats["saturation"] += 1
            survivors.remove(p)

    # --- Size outlier filter ---
    if use_size_outlier and not _stopped():
        for p, reason in find_size_outliers(survivors).items():
            if p not in to_delete:
                to_delete[p] = reason
                stats["size_outlier"] += 1
                if p in survivors:
                    survivors.remove(p)

    # --- Text-only page filter (parallel, limited workers for ML) ---
    if use_text_only and not _stopped():
        log.info("  [ML] Running CLIP classifier for text-only detection…")

        def _check_text_only(p: Path):
            img = load_image(p)
            if img:
                flagged, score = is_text_only_page(img)
                if flagged:
                    return p, f"text-only page (confidence={score:.0%})"
            return p, None

        ml_workers = min(2, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max(1, ml_workers)) as pool:
            futures = {pool.submit(_check_text_only, p): p for p in survivors}
            flagged_text = []
            for future in as_completed(futures):
                p, reason = future.result()
                if reason:
                    flagged_text.append((p, reason))
        for p, reason in flagged_text:
            to_delete[p] = reason
            stats["text_only"] += 1
            survivors.remove(p)

    # --- ML (CLIP) classifier — runs last (slowest, parallel with limited workers) ---
    if use_ml and not _stopped():
        # Only check first and last 20% or 20 pages (whichever is lower)
        total_survivors = len(survivors)
        if total_survivors > 0:
            # Calculate how many pages to check at each end
            pages_to_check = min(20, max(1, int(total_survivors * 0.2)))

            # Get indices to check (first N and last N pages)
            indices_to_check = set(range(pages_to_check))  # First N
            indices_to_check.update(range(max(0, total_survivors - pages_to_check), total_survivors))  # Last N

            candidates = [(i, survivors[i]) for i in sorted(indices_to_check)]

            log.info(
                f"  [ML] Running CLIP classifier on first and last {pages_to_check} pages (out of {total_survivors})…")

            def _check_ml(item):
                _i, p = item
                img = load_image(p)
                if img is None:
                    return p, None
                flagged, score = is_non_manga_ml(img)
                if flagged:
                    return p, f"ML: non-manga content (confidence={score:.0%})"
                return p, None

            # Use few workers for ML to avoid GPU/memory contention
            ml_workers = min(2, os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=max(1, ml_workers)) as pool:
                futures = {pool.submit(_check_ml, c): c for c in candidates}
                flagged_ml = []
                for future in as_completed(futures):
                    p, reason = future.result()
                    if reason:
                        flagged_ml.append((p, reason))
            for p, reason in flagged_ml:
                to_delete[p] = reason
                stats["ml_foreign"] += 1
                if p in survivors:
                    survivors.remove(p)

    if not to_delete:
        log.info("  ✅ Nothing to remove.\n")
        if collect_for_preview:
            stats["pending_delete"] = {}
            stats["all_images"] = image_files
        return stats

    # --- Preview table ---
    label = "[DRY RUN] " if dry_run else ""
    log.info(f"  {label}Flagged for removal ({len(to_delete)} file(s)):")
    for p, reason in sorted(to_delete.items(), key=lambda x: x[0].name):
        log.info(f"    - {p.name:<40}  {reason}")

    if dry_run:
        stats["skipped"] = len(to_delete)
        log.info(f"  ✅ Folder '{chapter_path.name}' analysis complete (dry run)\n")
        return stats

    # --- Collect mode: return flagged files for batch preview later ---
    if collect_for_preview:
        stats["pending_delete"] = to_delete
        stats["all_images"] = image_files
        log.info(f"  📋 Folder '{chapter_path.name}' — {len(to_delete)} file(s) queued for review\n")
        return stats

    # --- Confirm & delete ---
    file_list = "\n".join(
        f"  {p.name}  ({reason})"
        for p, reason in sorted(to_delete.items(), key=lambda x: x[0].name)
    )
    if confirm_callback is not None:
        confirmed = confirm_callback(chapter_path.name, file_list, len(to_delete))
    else:
        confirmed = messagebox.askyesno(
            "Confirm deletion",
            f"Delete {len(to_delete)} file(s) from '{chapter_path.name}'?\n\n{file_list}",
        )
    if not confirmed:
        log.info(f"  ⏭️ Folder '{chapter_path.name}' skipped by user.\n")
        stats["skipped"] = len(to_delete)
        return stats

    for p, reason in to_delete.items():
        try:
            p.unlink()
            log.info(f"  [DELETED] {p.name}  ({reason})")
            stats["removed"] += 1
        except Exception as exc:
            log.warning(f"  [ERROR] Could not delete {p.name}: {exc}")
            stats["skipped"] += 1

    log.info(f"  ✅ Folder '{chapter_path.name}' processing complete\n")
    return stats


# ---------------------------------------------------------------------------
# IMAGE PREVIEW DIALOG
# ---------------------------------------------------------------------------

class PreviewDialog(tk.Toplevel):
    """
    Modal dialog that shows each flagged image one-by-one and lets the user
    decide whether to delete or keep it.

    Optional smaller thumbnails of the previous and next images (neighbours
    in the folder) are shown on either side for context.

    Parameters:
        parent       — parent tkinter window
        root_name    — display name (e.g. manga root folder name)
        to_delete    — {Path: reason} of all flagged files across chapters
        all_images   — flat sorted list of ALL image Paths across chapters
                       (used to find neighbours for the context thumbnails)

    Keyboard shortcuts:
        D / Delete  — mark current image for deletion
        K / Right   — keep current image (skip)
        A           — mark ALL remaining images for deletion
        N           — keep ALL remaining images (skip all)
        Escape      — keep all remaining and close
        C           — toggle context thumbnails on/off

    Returns a dict of {Path: reason} for the files confirmed for deletion
    via get_confirmed().
    """

    MAX_PREVIEW_SIZE = 650  # max width/height for the main preview
    CONTEXT_THUMB_SIZE = 180  # max width/height for context thumbnails

    def __init__(self, parent, root_name: str, to_delete: dict, all_images: list[Path]):
        super().__init__(parent)
        self.title(f"Review flagged images — {root_name}")
        self.resizable(True, True)
        self.minsize(600, 500)
        self.transient(parent)
        self.grab_set()

        self._items = sorted(to_delete.items(), key=lambda x: x[0].name)
        self._index = 0
        self._confirmed: dict[Path, str] = {}
        self._history: list[str] = []  # tracks "delete" or "keep" per step for undo

        # Build a lookup for neighbour context: path → index in all_images
        self._all_images = all_images
        self._img_index: dict[Path, int] = {p: i for i, p in enumerate(all_images)}

        # PhotoImage refs to prevent GC
        self._photo_main = None
        self._photo_prev = None
        self._photo_next = None

        self._build_ui()
        self._bind_keys()
        self._show_current()

        # Center on parent
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        self.protocol("WM_DELETE_WINDOW", self._on_skip_all)
        self.wait_window(self)

    # ---- helpers ----

    @staticmethod
    def _load_thumbnail(path: Path, max_size: int):
        """Load an image and return a scaled PhotoImage, or None on failure."""
        try:
            img = Image.open(path)
            img.load()
            img = img.convert("RGB")
            w, h = img.size
            if w > max_size or h > max_size:
                ratio = min(max_size / w, max_size / h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    # ---- UI ----

    def _build_ui(self):
        # Top info bar
        info_frame = tk.Frame(self)
        info_frame.pack(fill="x", padx=10, pady=(10, 4))

        self.counter_label = tk.Label(
            info_frame, text="", font=("Segoe UI", 10, "bold"))
        self.counter_label.pack(side="left")

        self.filename_label = tk.Label(
            info_frame, text="", font=("Consolas", 10), fg="#2980b9")
        self.filename_label.pack(side="left", padx=(12, 0))

        self.folder_label = tk.Label(
            info_frame, text="", font=("Segoe UI", 9), fg="#888")
        self.folder_label.pack(side="right")

        # Reason
        self.reason_label = tk.Label(
            self, text="", font=("Segoe UI", 9), fg="#c0392b", wraplength=880,
            justify="left")
        self.reason_label.pack(fill="x", padx=10, pady=(0, 4))

        # ── Image area: [prev thumb] [main image] [next thumb] ──
        img_area = tk.Frame(self)
        img_area.pack(fill="both", expand=True, padx=10, pady=4)

        # Previous context thumbnail (left)
        self._prev_frame = tk.Frame(img_area, width=self.CONTEXT_THUMB_SIZE + 10)
        self._prev_frame.pack(side="left", fill="y", padx=(0, 6))
        self._prev_frame.pack_propagate(False)

        tk.Label(self._prev_frame, text="← prev", font=("Segoe UI", 7), fg="#888"
                 ).pack(side="top")
        self._prev_label = tk.Label(self._prev_frame, bg="#333", anchor="center")
        self._prev_label.pack(fill="both", expand=True)

        # Next context thumbnail (right)
        self._next_frame = tk.Frame(img_area, width=self.CONTEXT_THUMB_SIZE + 10)
        self._next_frame.pack(side="right", fill="y", padx=(6, 0))
        self._next_frame.pack_propagate(False)

        tk.Label(self._next_frame, text="next →", font=("Segoe UI", 7), fg="#888"
                 ).pack(side="top")
        self._next_label = tk.Label(self._next_frame, bg="#333", anchor="center")
        self._next_label.pack(fill="both", expand=True)

        # Main image (center)
        self.image_label = tk.Label(img_area, bg="#222", anchor="center")
        self.image_label.pack(fill="both", expand=True)

        # ── Context toggle ──
        toggle_frame = tk.Frame(self)
        toggle_frame.pack(fill="x", padx=10, pady=(2, 0))

        self._show_context_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toggle_frame,
            text="Show neighbouring pages for context  (C)",
            variable=self._show_context_var,
            command=self._toggle_context,
        ).pack(side="left")

        # ── Buttons ──
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=(4, 10))

        self.delete_btn = ttk.Button(
            btn_frame, text="🗑  Delete  (D)",
            command=self._on_delete)
        self.delete_btn.pack(side="left", padx=(0, 6))

        self.keep_btn = ttk.Button(
            btn_frame, text="✅  Keep  (K)",
            command=self._on_keep)
        self.keep_btn.pack(side="left", padx=(0, 6))

        self.back_btn = ttk.Button(
            btn_frame, text="⬅  Back  (B)",
            command=self._on_back, state="disabled")
        self.back_btn.pack(side="left", padx=(0, 6))

        ttk.Separator(btn_frame, orient="vertical").pack(
            side="left", fill="y", padx=8)

        self.delete_all_btn = ttk.Button(
            btn_frame, text="🗑  Delete ALL remaining  (A)",
            command=self._on_delete_all)
        self.delete_all_btn.pack(side="left", padx=(0, 6))

        self.skip_all_btn = ttk.Button(
            btn_frame, text="⏭  Keep ALL remaining  (N)",
            command=self._on_skip_all)
        self.skip_all_btn.pack(side="left")

        # Shortcut hint
        tk.Label(
            self,
            text="Shortcuts:  D / Del → delete  |  K / → → keep  |  B / ← → back  |  A → delete all  |  N / Esc → keep all  |  C → toggle context",
            font=("Segoe UI", 8), fg="#888",
        ).pack(side="bottom", pady=(0, 6))

    def _bind_keys(self):
        self.bind("<d>", lambda e: self._on_delete())
        self.bind("<D>", lambda e: self._on_delete())
        self.bind("<Delete>", lambda e: self._on_delete())
        self.bind("<k>", lambda e: self._on_keep())
        self.bind("<K>", lambda e: self._on_keep())
        self.bind("<Right>", lambda e: self._on_keep())
        self.bind("<b>", lambda e: self._on_back())
        self.bind("<B>", lambda e: self._on_back())
        self.bind("<Left>", lambda e: self._on_back())
        self.bind("<a>", lambda e: self._on_delete_all())
        self.bind("<A>", lambda e: self._on_delete_all())
        self.bind("<n>", lambda e: self._on_skip_all())
        self.bind("<N>", lambda e: self._on_skip_all())
        self.bind("<Escape>", lambda e: self._on_skip_all())
        self.bind("<c>", lambda e: self._toggle_context_key())
        self.bind("<C>", lambda e: self._toggle_context_key())

    def _toggle_context_key(self):
        """Toggle the context checkbox via keyboard."""
        self._show_context_var.set(not self._show_context_var.get())
        self._toggle_context()

    def _toggle_context(self):
        """Show or hide the prev/next context thumbnail panels."""
        if self._show_context_var.get():
            self._prev_frame.pack(side="left", fill="y", padx=(0, 6),
                                  before=self.image_label)
            self._next_frame.pack(side="right", fill="y", padx=(6, 0),
                                  before=self.image_label)
        else:
            self._prev_frame.pack_forget()
            self._next_frame.pack_forget()

    # ---- Navigation ----

    def _get_neighbour(self, path: Path, offset: int):
        """
        Return the Path of the image *offset* positions away from *path*
        in the all_images list, or None if out of bounds.
        """
        idx = self._img_index.get(path)
        if idx is None:
            return None
        target = idx + offset
        if 0 <= target < len(self._all_images):
            return self._all_images[target]
        return None

    def _show_current(self):
        if self._index >= len(self._items):
            self.destroy()
            return

        path, reason = self._items[self._index]
        total = len(self._items)

        self.counter_label.config(text=f"[{self._index + 1} / {total}]")
        self.filename_label.config(text=path.name)
        self.folder_label.config(text=f"📁 {path.parent.name}")
        self.reason_label.config(text=f"Reason: {reason}")

        # --- Main preview ---
        self._photo_main = self._load_thumbnail(path, self.MAX_PREVIEW_SIZE)
        if self._photo_main:
            self.image_label.config(image=self._photo_main, text="")
        else:
            self.image_label.config(image="", text="⚠ Could not load preview",
                                    fg="white", font=("Segoe UI", 12))

        # --- Context thumbnails ---
        prev_path = self._get_neighbour(path, -1)
        next_path = self._get_neighbour(path, +1)

        if prev_path:
            self._photo_prev = self._load_thumbnail(prev_path, self.CONTEXT_THUMB_SIZE)
            if self._photo_prev:
                self._prev_label.config(image=self._photo_prev, text="")
            else:
                self._prev_label.config(image="", text="—", fg="#888")
        else:
            self._photo_prev = None
            self._prev_label.config(image="", text="(start)", fg="#888")

        if next_path:
            self._photo_next = self._load_thumbnail(next_path, self.CONTEXT_THUMB_SIZE)
            if self._photo_next:
                self._next_label.config(image=self._photo_next, text="")
            else:
                self._next_label.config(image="", text="—", fg="#888")
        else:
            self._photo_next = None
            self._next_label.config(image="", text="(end)", fg="#888")

        self.focus_set()

    def _advance(self):
        self._index += 1
        self._update_back_btn()
        self._show_current()

    def _update_back_btn(self):
        """Enable the Back button only when there is history to undo."""
        if self._history:
            self.back_btn.config(state="normal")
        else:
            self.back_btn.config(state="disabled")

    # ---- Actions ----

    def _on_delete(self):
        if self._index < len(self._items):
            path, reason = self._items[self._index]
            self._confirmed[path] = reason
            self._history.append("delete")
        self._advance()

    def _on_keep(self):
        if self._index < len(self._items):
            self._history.append("keep")
        self._advance()

    def _on_back(self):
        """Go back one image and undo the previous decision."""
        if not self._history or self._index <= 0:
            return
        self._index -= 1
        action = self._history.pop()
        # Undo: if the previous action was "delete", remove it from confirmed
        if action == "delete":
            path, _reason = self._items[self._index]
            self._confirmed.pop(path, None)
        self._update_back_btn()
        self._show_current()

    def _on_delete_all(self):
        for i in range(self._index, len(self._items)):
            path, reason = self._items[i]
            self._confirmed[path] = reason
        self.destroy()

    def _on_skip_all(self):
        # Keep all remaining — don't add them to confirmed
        self.destroy()

    def get_confirmed(self) -> dict:
        return self._confirmed


# ---------------------------------------------------------------------------
# TOOLTIP HELPER
# ---------------------------------------------------------------------------

class ToolTip:
    """
    Lightweight tooltip that appears when the mouse hovers over a widget.
    Usage:  ToolTip(widget, "Helpful description text")
    """

    DELAY_MS = 400  # delay before showing
    WRAP_PX = 350  # max line width in pixels

    def __init__(self, widget, text: str):
        self._widget = widget
        self._text = text
        self._tip_window = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._cancel, add="+")
        widget.bind("<ButtonPress>", self._cancel, add="+")

    def _schedule(self, _event):
        self._cancel()
        self._after_id = self._widget.after(self.DELAY_MS, self._show)

    def _cancel(self, _event=None):
        if self._after_id:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide()

    def _show(self):
        if self._tip_window:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        label = tk.Label(
            tw, text=self._text, justify="left",
            background="#ffffe0", foreground="#333",
            relief="solid", borderwidth=1,
            wraplength=self.WRAP_PX,
            font=("Segoe UI", 9),
            padx=6, pady=4,
        )
        label.pack()
        self._tip_window = tw

    def _hide(self):
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class CleanerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Manga / Image Folder Cleaner")
        self.resizable(True, True)
        self.minsize(700, 520)
        self._build_ui()
        # Redirect the module-level logger to the GUI log area
        self._attach_log_handler()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        PAD = 10

        # ── Top: folder list ──────────────────────────────────────────
        folder_frame = ttk.LabelFrame(self,
                                      text="Folders to scan (recursively searches up to 4 levels deep for folders with images)")
        folder_frame.pack(fill="both", expand=False, padx=PAD, pady=(PAD, 0))

        list_frame = tk.Frame(folder_frame)
        list_frame.pack(fill="both", expand=True, padx=6, pady=6)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.folder_listbox = tk.Listbox(
            list_frame, selectmode="extended", height=6,
            yscrollcommand=scrollbar.set, activestyle="none",
            relief="flat", bd=1, highlightthickness=1,
            highlightbackground="#aaa", font=("Consolas", 9),
        )
        scrollbar.config(command=self.folder_listbox.yview)
        self.folder_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        btn_row = tk.Frame(folder_frame)
        btn_row.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(btn_row, text="➕  Add folder(s)", command=self._add_folders).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="➖  Remove selected", command=self._remove_selected).pack(side="left")

        # ── Middle: options ───────────────────────────────────────────
        opt_frame = ttk.LabelFrame(self, text="Options")
        opt_frame.pack(fill="x", padx=PAD, pady=(6, 0))

        self.dry_run_var = tk.BooleanVar(value=DRY_RUN)
        dry_cb = ttk.Checkbutton(
            opt_frame, text="Dry-run mode  (preview only — no files will be deleted)",
            variable=self.dry_run_var,
        )
        dry_cb.grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ToolTip(dry_cb,
                "When enabled, the cleaner will only scan and log what it finds "
                "without deleting any files. Use this to review results before "
                "committing to deletions. Disable to allow actual file removal.")

        self.preview_var = tk.BooleanVar(value=True)
        preview_cb = ttk.Checkbutton(
            opt_frame,
            text="Preview images before delete  (review each flagged image — only when dry-run is OFF)",
            variable=self.preview_var,
        )
        preview_cb.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 4))
        ToolTip(preview_cb,
                "When dry-run is OFF, this opens a preview window showing each "
                "flagged image one by one. You can mark each image for deletion "
                "or keep it, with context thumbnails of neighbouring pages. "
                "The preview appears after all folders of each root are scanned.")

        # Threshold controls
        thresh_row = tk.Frame(opt_frame)
        thresh_row.grid(row=2, column=0, sticky="w", padx=8, pady=(0, 6))

        def _labeled_spin(parent, label, default, lo, hi, tooltip=None):
            lbl = tk.Label(parent, text=label)
            lbl.pack(side="left")
            var = tk.IntVar(value=default)
            spin = ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=5)
            spin.pack(side="left", padx=(2, 12))
            if tooltip:
                ToolTip(lbl, tooltip)
                ToolTip(spin, tooltip)
            return var

        self.white_var = _labeled_spin(thresh_row, "White ≥", WHITE_THRESHOLD, 200, 255,
                                       "Pixel brightness (0–255) at or above which a pixel counts as 'white'. "
                                       "A page is flagged as blank-white when ≥ Solid Ratio of its pixels "
                                       "meet this threshold. Lower = more aggressive detection.")
        self.black_var = _labeled_spin(thresh_row, "Black ≤", BLACK_THRESHOLD, 0, 50,
                                       "Pixel brightness (0–255) at or below which a pixel counts as 'black'. "
                                       "A page is flagged as blank-black when ≥ Solid Ratio of its pixels "
                                       "meet this threshold. Higher = more aggressive detection.")
        self.hash_var = _labeled_spin(thresh_row, "Hash dist ≤", HASH_THRESHOLD, 0, 64,
                                      "Maximum perceptual hash (pHash) distance between two images to "
                                      "consider them near-duplicates. 0 = only identical hashes match. "
                                      "8–10 is a good balance for manga. Higher = catches more duplicates "
                                      "but risks false positives.")

        solid_row = tk.Frame(opt_frame)
        solid_row.grid(row=3, column=0, sticky="w", padx=8, pady=(0, 8))
        solid_lbl = tk.Label(solid_row, text="Solid ratio ≥")
        solid_lbl.pack(side="left")
        self.solid_var = tk.DoubleVar(value=SOLID_RATIO)
        solid_spin = ttk.Spinbox(solid_row, from_=0.5, to=1.0, increment=0.01,
                                 textvariable=self.solid_var, width=6, format="%.2f")
        solid_spin.pack(side="left", padx=(2, 0))
        solid_tip = ("Fraction of pixels that must be solid white or solid black "
                     "to flag a page as blank. 0.99 means 99% of the page must "
                     "be a single solid colour. Lower = more aggressive, but "
                     "may flag pages with a small amount of content.")
        ToolTip(solid_lbl, solid_tip)
        ToolTip(solid_spin, solid_tip)

        # ── Advanced detection ────────────────────────────────────────
        adv_frame = ttk.LabelFrame(self,
                                   text="Advanced Detection  (optional — may produce false positives, review with dry-run first)")
        adv_frame.pack(fill="x", padx=PAD, pady=(6, 0))

        def _adv_row(parent, row, label, bool_var, widgets_fn=None, tooltip=None):
            """Helper: checkbox + optional inline controls on the same row."""
            cb = ttk.Checkbutton(parent, text=label, variable=bool_var)
            cb.grid(row=row, column=0, sticky="w", padx=8, pady=3)
            if tooltip:
                ToolTip(cb, tooltip)
            if widgets_fn:
                inner = tk.Frame(parent)
                inner.grid(row=row, column=1, sticky="w", padx=(0, 8), pady=3)
                widgets_fn(inner)

        def _spin_pair(parent, label, var, lo, hi, inc=1, fmt=None, width=6, tooltip=None):
            lbl = tk.Label(parent, text=label)
            lbl.pack(side="left")
            kw = dict(from_=lo, to=hi, textvariable=var, width=width)
            if inc != 1:
                kw["increment"] = inc
            if fmt:
                kw["format"] = fmt
            spin = ttk.Spinbox(parent, **kw)
            spin.pack(side="left", padx=(2, 10))
            if tooltip:
                ToolTip(lbl, tooltip)
                ToolTip(spin, tooltip)

        # Aspect ratio
        self.use_aspect_var = tk.BooleanVar(value=False)
        self.aspect_min_var = tk.DoubleVar(value=ASPECT_RATIO_MIN)
        self.aspect_max_var = tk.DoubleVar(value=ASPECT_RATIO_MAX)

        def _aspect_widgets(f):
            _spin_pair(f, "Min ratio", self.aspect_min_var, 0.1, 1.0, 0.01, "%.2f",
                       tooltip="Minimum acceptable width/height ratio. "
                               "Pages narrower than this are flagged. "
                               "Typical manga portrait page ≈ 0.55–0.80.")
            _spin_pair(f, "Max ratio", self.aspect_max_var, 0.1, 2.0, 0.01, "%.2f",
                       tooltip="Maximum acceptable width/height ratio. "
                               "Pages wider than this (landscape, square) are flagged. "
                               "Typical manga portrait page ≈ 0.55–0.80.")

        _adv_row(adv_frame, 0,
                 "🔲  Aspect ratio filter  (flag landscape / square pages)",
                 self.use_aspect_var, _aspect_widgets,
                 tooltip="Flag images whose width/height ratio falls outside the "
                         "expected manga portrait range. Catches landscape fan-art, "
                         "square ad images, and other non-standard page dimensions. "
                         "Adjust Min/Max ratio to match your manga's typical proportions.")

        # Saturation
        self.use_sat_var = tk.BooleanVar(value=False)
        self.sat_thresh_var = tk.DoubleVar(value=SATURATION_THRESHOLD)

        def _sat_widgets(f):
            _spin_pair(f, "Saturation threshold", self.sat_thresh_var, 0.0, 1.0, 0.01, "%.2f",
                       tooltip="Average HSV saturation (0.0–1.0) above which a page is "
                               "considered too colorful for B&W manga. Lower = more "
                               "sensitive. 0.18 works well for typical manga vs. colour inserts.")

        _adv_row(adv_frame, 1,
                 "🎨  Saturation filter  (flag colorful fan-art / color inserts)",
                 self.use_sat_var, _sat_widgets,
                 tooltip="Manga pages are mostly black & white with low colour saturation. "
                         "This filter flags images with high average colour saturation, "
                         "which are likely fan-art, colour inserts, or advertisements. "
                         "May flag intentionally coloured manga pages (covers, colour chapters).")

        # Size outlier
        self.use_size_var = tk.BooleanVar(value=False)
        self.size_z_var = tk.DoubleVar(value=SIZE_OUTLIER_Z)

        def _size_widgets(f):
            _spin_pair(f, "Z-score threshold", self.size_z_var, 1.0, 6.0, 0.1, "%.1f",
                       tooltip="How many standard deviations from the mean pixel area an "
                               "image must be to be considered an outlier. Lower = more "
                               "sensitive. 2.5 catches obviously different resolutions "
                               "without flagging minor variations.")

        _adv_row(adv_frame, 2,
                 "📐  Size outlier filter  (flag pages with very different resolution)",
                 self.use_size_var, _size_widgets,
                 tooltip="Flags images whose pixel area (width × height) is a statistical "
                         "outlier compared to other pages in the same folder. "
                         "Catches foreign inserts scanned at a completely different "
                         "resolution. Requires at least 4 images per folder to work.")

        # Text-only
        self.use_text_var = tk.BooleanVar(value=False)

        def _text_widgets(f):
            tk.Label(f, text="(uses CLIP ML model — requires: pip install torch transformers)",
                     font=("Segoe UI", 8), fg="#888").pack(side="left")

        _adv_row(adv_frame, 3,
                 "📄  Text-only filter  (flag surveys, TOC, afterword pages)",
                 self.use_text_var, _text_widgets,
                 tooltip="Uses the CLIP machine learning model to detect pages that "
                         "contain only text and no artwork — such as table of contents, "
                         "survey forms, author notes, afterword, or credits pages. "
                         "Requires PyTorch and Transformers to be installed. "
                         "The model is downloaded on first use (~350 MB).")

        # ML (CLIP)
        self.use_ml_var = tk.BooleanVar(value=False)
        self.ml_conf_var = tk.DoubleVar(value=ML_CONFIDENCE_THRESHOLD)

        def _ml_widgets(f):
            _spin_pair(f, "Confidence ≥", self.ml_conf_var, 0.5, 1.0, 0.01, "%.2f",
                       tooltip="Minimum confidence score (0.0–1.0) that an image is NOT "
                               "a manga page before flagging it. Higher = fewer false "
                               "positives but may miss some non-manga content. "
                               "0.75–0.85 is a good range.")
            tk.Label(f, text="(requires: pip install torch transformers)",
                     font=("Segoe UI", 8), fg="#888").pack(side="left")

        _adv_row(adv_frame, 4,
                 "🤖  ML classifier  (CLIP zero-shot — detects fan-art / ads)",
                 self.use_ml_var, _ml_widgets,
                 tooltip="Uses OpenAI's CLIP model for zero-shot image classification "
                         "to detect non-manga content such as fan-art, advertisements, "
                         "and promotional illustrations. Only scans the first and last "
                         "20% (or 20 pages, whichever is lower) of each folder, since "
                         "non-manga inserts typically appear at the beginning or end. "
                         "Requires PyTorch and Transformers.")

        # ── Run button ────────────────────────────────────────────────
        run_row = tk.Frame(self)
        run_row.pack(fill="x", padx=PAD, pady=6)
        self.run_btn = ttk.Button(run_row, text="▶  Run Cleaner", command=self._run, style="Accent.TButton")
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(run_row, text="⏹  Stop", command=self._stop)
        # stop button is hidden until processing starts
        self.status_label = tk.Label(run_row, text="", fg="#555", font=("Segoe UI", 9))
        self.status_label.pack(side="left", padx=12)

        self._stop_event = threading.Event()

        # ── Bottom: log output ────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="Log output")
        log_frame.pack(fill="both", expand=True, padx=PAD, pady=(0, PAD))

        log_inner = tk.Frame(log_frame)
        log_inner.pack(fill="both", expand=True, padx=6, pady=6)

        log_scroll_y = ttk.Scrollbar(log_inner, orient="vertical")
        log_scroll_x = ttk.Scrollbar(log_inner, orient="horizontal")
        self.log_text = tk.Text(
            log_inner, state="disabled", wrap="none", height=12,
            font=("Consolas", 9), relief="flat", bd=1,
            yscrollcommand=log_scroll_y.set,
            xscrollcommand=log_scroll_x.set,
        )
        log_scroll_y.config(command=self.log_text.yview)
        log_scroll_x.config(command=self.log_text.xview)
        log_scroll_x.pack(side="bottom", fill="x")
        log_scroll_y.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        # Tag colours
        self.log_text.tag_config("deleted", foreground="#c0392b")
        self.log_text.tag_config("ok", foreground="#27ae60")
        self.log_text.tag_config("warn", foreground="#e67e22")
        self.log_text.tag_config("header", foreground="#2980b9", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("dryrun", foreground="#8e44ad")

    # ------------------------------------------------------------------
    # Folder list management
    # ------------------------------------------------------------------

    def _add_folders(self):
        folder = filedialog.askdirectory(title="Select a root folder to add")
        if folder and folder not in self.folder_listbox.get(0, "end"):
            self.folder_listbox.insert("end", folder)

    def _remove_selected(self):
        for idx in reversed(self.folder_listbox.curselection()):
            self.folder_listbox.delete(idx)

    # ------------------------------------------------------------------
    # Logging bridge
    # ------------------------------------------------------------------

    def _attach_log_handler(self):
        handler = _TextHandler(self._append_log)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)

    def _append_log(self, message: str):
        # If called from a background thread, reschedule on the main thread
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda: self._append_log(message))
            return

        self.log_text.config(state="normal")
        # Pick a colour tag based on message content
        tag = ""
        lower = message.lower()
        if "[deleted]" in lower:
            tag = "deleted"
        elif "✅" in message or "success" in lower:
            tag = "ok"
        elif "[warn]" in lower or "[error]" in lower:
            tag = "warn"
        elif message.startswith("📁"):
            tag = "header"
        elif "[dry run]" in lower:
            tag = "dryrun"

        self.log_text.insert("end", message + "\n", tag)
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.update_idletasks()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self):
        folders = list(self.folder_listbox.get(0, "end"))
        if not folders:
            messagebox.showwarning("No folders", "Please add at least one folder.")
            return

        # Apply UI values to module-level thresholds so all helpers pick them up
        global WHITE_THRESHOLD, BLACK_THRESHOLD, SOLID_RATIO, HASH_THRESHOLD
        global ASPECT_RATIO_MIN, ASPECT_RATIO_MAX, SATURATION_THRESHOLD
        global SIZE_OUTLIER_Z, ML_CONFIDENCE_THRESHOLD
        WHITE_THRESHOLD = self.white_var.get()
        BLACK_THRESHOLD = self.black_var.get()
        SOLID_RATIO = self.solid_var.get()
        HASH_THRESHOLD = self.hash_var.get()
        ASPECT_RATIO_MIN = self.aspect_min_var.get()
        ASPECT_RATIO_MAX = self.aspect_max_var.get()
        SATURATION_THRESHOLD = self.sat_thresh_var.get()
        SIZE_OUTLIER_Z = self.size_z_var.get()
        ML_CONFIDENCE_THRESHOLD = self.ml_conf_var.get()

        # Snapshot all options before launching the thread
        self._work_opts = {
            "folders": folders,
            "dry_run": self.dry_run_var.get(),
            "use_preview": self.preview_var.get() and not self.dry_run_var.get(),
            "use_aspect": self.use_aspect_var.get(),
            "use_sat": self.use_sat_var.get(),
            "use_size": self.use_size_var.get(),
            "use_text": self.use_text_var.get(),
            "use_ml": self.use_ml_var.get(),
        }

        # Clear log
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

        self._stop_event.clear()
        self.run_btn.config(state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))
        self.status_label.config(text="Running…")
        self.update_idletasks()

        # Launch processing on a background thread so the GUI stays responsive
        worker = threading.Thread(target=self._run_worker, daemon=True)
        worker.start()

    def _stop(self):
        """Signal the worker thread to stop after the current chapter."""
        self._stop_event.set()
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Stopping… (will finish current folder)")

    # ------------------------------------------------------------------
    # Thread-safe GUI helpers  (called from worker, run on main thread)
    # ------------------------------------------------------------------

    def _show_preview_from_worker(self, root_name, root_pending, root_all_images):
        """
        Open the PreviewDialog on the main thread and block the worker
        until the user finishes reviewing.  Returns the confirmed dict.
        """
        result = [{}]
        event = threading.Event()

        def _open():
            dlg = PreviewDialog(self, root_name, root_pending, root_all_images)
            result[0] = dlg.get_confirmed()
            event.set()

        self.after(0, _open)
        event.wait()
        return result[0]

    def _confirm_delete_from_worker(self, chapter_name, file_list_text, count):
        """
        Show a messagebox.askyesno on the main thread and block the worker
        until the user responds.  Returns True/False.
        """
        result = [False]
        event = threading.Event()

        def _ask():
            result[0] = messagebox.askyesno(
                "Confirm deletion",
                f"Delete {count} file(s) from '{chapter_name}'?\n\n{file_list_text}",
            )
            event.set()

        self.after(0, _ask)
        event.wait()
        return result[0]

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _run_worker(self):
        """Heavy processing — runs on a background thread."""
        opts = self._work_opts
        folders = opts["folders"]
        dry_run = opts["dry_run"]
        use_preview = opts["use_preview"]
        use_aspect = opts["use_aspect"]
        use_sat = opts["use_sat"]
        use_size = opts["use_size"]
        use_text = opts["use_text"]
        use_ml = opts["use_ml"]

        totals = {
            "scanned": 0, "blank": 0, "duplicate": 0,
            "aspect": 0, "saturation": 0, "size_outlier": 0,
            "text_only": 0, "ml_foreign": 0,
            "removed": 0, "skipped": 0,
        }
        total_chapters = 0

        active_filters = []
        if use_aspect:   active_filters.append("aspect")
        if use_sat:      active_filters.append("saturation")
        if use_size:     active_filters.append("size-outlier")
        if use_text:     active_filters.append("text-only")
        if use_ml:       active_filters.append("ML/CLIP")
        filter_label = "  |  filters: " + ", ".join(active_filters) if active_filters else ""

        mode_label = "DRY RUN" if dry_run else "LIVE MODE"
        self._append_log(f"{'=' * 60}")
        self._append_log(f"  {mode_label}  |  white≥{WHITE_THRESHOLD}  black≤{BLACK_THRESHOLD}  "
                         f"solid≥{SOLID_RATIO}  hash≤{HASH_THRESHOLD}{filter_label}")
        self._append_log(f"{'=' * 60}")

        try:
            for root_str in folders:
                if self._stop_event.is_set():
                    self._append_log("\n⏹ Processing stopped by user.")
                    break

                root = Path(root_str)
                if not root.is_dir():
                    self._append_log(f"[WARN] Not a valid directory, skipping: {root_str}")
                    continue

                # Recursively find all folders with images (up to 4 levels deep)
                image_dirs = find_image_folders(root)
                self._append_log(f"\n📂 Root: {root}  ({len(image_dirs)} folder(s) with images)")

                if not image_dirs:
                    self._append_log("  (no folders with images found)")
                    continue

                # Accumulate flagged files across all chapters for batch preview
                root_pending: dict[Path, str] = {}  # path → reason
                root_all_images: list[Path] = []  # all images across chapters (for context)

                for chapter in image_dirs:
                    if self._stop_event.is_set():
                        self._append_log("\n⏹ Processing stopped by user.")
                        break

                    # Show relative path from root for nested folders
                    try:
                        rel = chapter.relative_to(root)
                        display = str(rel) if str(rel) != "." else root.name
                    except ValueError:
                        display = chapter.name
                    self._append_log(f"📁 Processing: {display}")

                    stats = process_chapter(
                        chapter, dry_run,
                        use_aspect=use_aspect,
                        use_saturation=use_sat,
                        use_size_outlier=use_size,
                        use_text_only=use_text,
                        use_ml=use_ml,
                        collect_for_preview=use_preview,
                        confirm_callback=self._confirm_delete_from_worker,
                        stop_event=self._stop_event,
                    )

                    # Accumulate pending items when in preview mode
                    if use_preview:
                        pending = stats.pop("pending_delete", {})
                        all_imgs = stats.pop("all_images", [])
                        root_pending.update(pending)
                        root_all_images.extend(all_imgs)

                    for k in totals:
                        totals[k] += stats.get(k, 0)
                    total_chapters += 1

                # --- After all chapters: show one batch preview for this root ---
                if use_preview and root_pending and not self._stop_event.is_set():
                    self._append_log(
                        f"\n🖼  Opening preview for {len(root_pending)} flagged image(s) from '{root.name}'…")

                    confirmed = self._show_preview_from_worker(
                        root.name, root_pending, root_all_images)

                    skipped_count = len(root_pending) - len(confirmed)
                    totals["skipped"] += skipped_count

                    if not confirmed:
                        self._append_log(f"  ⏭️ All files kept by user for '{root.name}'.\n")
                    else:
                        for p, reason in confirmed.items():
                            try:
                                p.unlink()
                                log.info(f"  [DELETED] {p.name}  ({reason})")
                                totals["removed"] += 1
                            except Exception as exc:
                                log.warning(f"  [ERROR] Could not delete {p.name}: {exc}")
                                totals["skipped"] += 1
                        self._append_log(f"  ✅ '{root.name}' — deleted {len(confirmed)}, kept {skipped_count}\n")

            # Summary
            total_flagged = sum(totals[k] for k in ("blank", "duplicate", "aspect",
                                                    "saturation", "size_outlier",
                                                    "text_only", "ml_foreign"))
            self._append_log(f"\n{'=' * 60}")
            self._append_log("  SUMMARY")
            self._append_log(f"{'=' * 60}")
            self._append_log(f"  Folders processed     : {total_chapters}")
            self._append_log(f"  Images scanned        : {totals['scanned']}")
            self._append_log(f"  Blank pages           : {totals['blank']}")
            self._append_log(f"  Duplicates            : {totals['duplicate']}")
            if use_aspect:   self._append_log(f"  Wrong aspect ratio    : {totals['aspect']}")
            if use_sat:      self._append_log(f"  High saturation       : {totals['saturation']}")
            if use_size:     self._append_log(f"  Size outliers         : {totals['size_outlier']}")
            if use_text:     self._append_log(f"  Text-only pages       : {totals['text_only']}")
            if use_ml:       self._append_log(f"  ML non-manga          : {totals['ml_foreign']}")
            if dry_run:
                self._append_log(f"  Would be deleted      : {total_flagged}")
            else:
                self._append_log(f"  Deleted               : {totals['removed']}")
                self._append_log(f"  Skipped / errors      : {totals['skipped']}")
            self._append_log(f"\n  Full log saved to: {LOG_FILE}")
            self._append_log(f"{'=' * 60}")

        except Exception as exc:
            self._append_log(f"[ERROR] Unexpected error: {exc}")

        # Re-enable the Run button on the main thread
        self.after(0, self._run_finished)

    def _run_finished(self):
        """Called on the main thread when the worker thread is done."""
        self.run_btn.config(state="normal")
        self.stop_btn.pack_forget()
        self.stop_btn.config(state="normal")
        stopped = self._stop_event.is_set()
        self.status_label.config(text="Stopped ⏹" if stopped else "Done ✅")


# Tiny logging handler that forwards records to a callable
class _TextHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__()
        self._cb = callback

    def emit(self, record):
        self._cb(self.format(record))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    app = CleanerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
