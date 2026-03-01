"""
Chapter processor for Manga Cleaner.

Scans a single chapter folder, runs all enabled detection passes (blank,
duplicate, aspect ratio, saturation, size outlier, text-only, ML), and
either deletes the flagged files or collects them for batch preview.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tkinter import messagebox

from manga_cleaner.config import config
from manga_cleaner.detection import (
    find_size_outliers,
    is_high_saturation,
    is_wrong_aspect_ratio,
)
from manga_cleaner.image_helpers import find_duplicates, is_blank_page, load_image
from manga_cleaner.logging_setup import log
from manga_cleaner.ml_classifier import is_non_manga_ml, is_text_only_page


def process_chapter(
    chapter_path: Path,
    dry_run: bool,
    *,
    use_aspect: bool = False,
    use_saturation: bool = False,
    use_size_outlier: bool = False,
    use_text_only: bool = False,
    use_ml: bool = False,
    collect_for_preview: bool = False,
    confirm_callback=None,
    stop_event: threading.Event | None = None,
) -> dict:
    """Scan one chapter folder and detect unwanted images.

    Parameters
    ----------
    chapter_path : Path
        Directory containing image files for a single chapter / volume.
    dry_run : bool
        If ``True``, only log what would be deleted.
    use_aspect, use_saturation, use_size_outlier, use_text_only, use_ml : bool
        Enable the corresponding advanced detection pass.
    collect_for_preview : bool
        If ``True``, skip deletion and return flagged files in
        ``stats["pending_delete"]`` and all images in ``stats["all_images"]``.
    confirm_callback : callable, optional
        ``(chapter_name, file_list_text, count) -> bool`` used to show a
        confirmation dialog from the GUI thread.
    stop_event : threading.Event, optional
        If set, the function returns early with partial results.

    Returns
    -------
    dict
        Statistics: *scanned*, *blank*, *duplicate*, *aspect*, *saturation*,
        *size_outlier*, *text_only*, *ml_foreign*, *removed*, *skipped*,
        and optionally *pending_delete* / *all_images*.
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
        if p.is_file() and p.suffix.lower() in config.supported_extensions
    )

    if not image_files:
        log.info("  (no images found)\n")
        return stats

    stats["scanned"] = len(image_files)

    def _stopped():
        return stop_event is not None and stop_event.is_set()

    to_delete: dict[Path, str] = {}
    non_blank: list[Path] = []

    # --- Blank-page detection (parallel) ------------------------------------
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
    non_blank.sort(key=lambda x: x.name)

    if _stopped():
        return stats

    # --- Duplicate detection ------------------------------------------------
    for p, reason in find_duplicates(non_blank).items():
        if reason == "corrupted":
            to_delete[p] = "corrupted"
        else:
            to_delete[p] = reason
            stats["duplicate"] += 1

    if _stopped():
        return stats

    survivors = [p for p in non_blank if p not in to_delete]

    # --- Aspect ratio (parallel) --------------------------------------------
    if use_aspect and not _stopped():
        def _check_aspect(p: Path):
            img = load_image(p)
            if img and is_wrong_aspect_ratio(img):
                return p, (
                    f"wrong aspect ratio ({img.width}x{img.height}, "
                    f"ratio={img.width / img.height:.2f})"
                )
            return p, None

        with ThreadPoolExecutor(
            max_workers=max(1, min(os.cpu_count() or 4, len(survivors), 8))
        ) as pool:
            futures = {pool.submit(_check_aspect, p): p for p in survivors}
            flagged = []
            for future in as_completed(futures):
                p, reason = future.result()
                if reason:
                    flagged.append((p, reason))
        for p, reason in flagged:
            to_delete[p] = reason
            stats["aspect"] += 1
            survivors.remove(p)

    # --- Saturation (parallel) ----------------------------------------------
    if use_saturation and not _stopped():
        def _check_sat(p: Path):
            img = load_image(p)
            if img and is_high_saturation(img):
                return p, "high color saturation (likely fan-art / color insert)"
            return p, None

        with ThreadPoolExecutor(
            max_workers=max(1, min(os.cpu_count() or 4, len(survivors), 8))
        ) as pool:
            futures = {pool.submit(_check_sat, p): p for p in survivors}
            flagged = []
            for future in as_completed(futures):
                p, reason = future.result()
                if reason:
                    flagged.append((p, reason))
        for p, reason in flagged:
            to_delete[p] = reason
            stats["saturation"] += 1
            survivors.remove(p)

    # --- Size outlier -------------------------------------------------------
    if use_size_outlier and not _stopped():
        for p, reason in find_size_outliers(survivors).items():
            if p not in to_delete:
                to_delete[p] = reason
                stats["size_outlier"] += 1
                if p in survivors:
                    survivors.remove(p)

    # --- Text-only (ML) -----------------------------------------------------
    if use_text_only and not _stopped():
        log.info("  [ML] Running CLIP classifier for text-only detection...")

        def _check_text(p: Path):
            img = load_image(p)
            if img:
                flagged, score = is_text_only_page(img)
                if flagged:
                    return p, f"text-only page (confidence={score:.0%})"
            return p, None

        ml_w = min(2, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max(1, ml_w)) as pool:
            futures = {pool.submit(_check_text, p): p for p in survivors}
            flagged = []
            for future in as_completed(futures):
                p, reason = future.result()
                if reason:
                    flagged.append((p, reason))
        for p, reason in flagged:
            to_delete[p] = reason
            stats["text_only"] += 1
            survivors.remove(p)

    # --- ML / CLIP ----------------------------------------------------------
    if use_ml and not _stopped():
        total_survivors = len(survivors)
        if total_survivors > 0:
            n = min(20, max(1, int(total_survivors * 0.2)))
            idx_set = set(range(n))
            idx_set.update(range(max(0, total_survivors - n), total_survivors))
            candidates = [(i, survivors[i]) for i in sorted(idx_set)]

            log.info(
                f"  [ML] Running CLIP classifier on first and last "
                f"{n} pages (out of {total_survivors})..."
            )

            def _check_ml(item):
                _i, p = item
                img = load_image(p)
                if img is None:
                    return p, None
                flagged, score = is_non_manga_ml(img)
                if flagged:
                    return p, f"ML: non-manga content (confidence={score:.0%})"
                return p, None

            ml_w = min(2, os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=max(1, ml_w)) as pool:
                futures = {pool.submit(_check_ml, c): c for c in candidates}
                flagged = []
                for future in as_completed(futures):
                    p, reason = future.result()
                    if reason:
                        flagged.append((p, reason))
            for p, reason in flagged:
                to_delete[p] = reason
                stats["ml_foreign"] += 1
                if p in survivors:
                    survivors.remove(p)

    # --- Nothing to delete --------------------------------------------------
    if not to_delete:
        log.info("  ✅ Nothing to remove.\n")
        if collect_for_preview:
            stats["pending_delete"] = {}
            stats["all_images"] = image_files
        return stats

    # --- Log flagged files --------------------------------------------------
    lbl = "[DRY RUN] " if dry_run else ""
    log.info(f"  {lbl}Flagged for removal ({len(to_delete)} file(s)):")
    for p, reason in sorted(to_delete.items(), key=lambda x: x[0].name):
        log.info(f"    - {p.name:<40}  {reason}")

    if dry_run:
        stats["skipped"] = len(to_delete)
        log.info(f"  ✅ Folder '{chapter_path.name}' analysis complete (dry run)\n")
        return stats

    # --- Collect mode -------------------------------------------------------
    if collect_for_preview:
        stats["pending_delete"] = to_delete
        stats["all_images"] = image_files
        log.info(
            f"  📋 Folder '{chapter_path.name}' — "
            f"{len(to_delete)} file(s) queued for review\n"
        )
        return stats

    # --- Confirm & delete ---------------------------------------------------
    file_list = "\n".join(
        f"  {p.name}  ({reason})"
        for p, reason in sorted(to_delete.items(), key=lambda x: x[0].name)
    )
    if confirm_callback is not None:
        confirmed = confirm_callback(chapter_path.name, file_list, len(to_delete))
    else:
        confirmed = messagebox.askyesno(
            "Confirm deletion",
            f"Delete {len(to_delete)} file(s) from "
            f"'{chapter_path.name}'?\n\n{file_list}",
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

