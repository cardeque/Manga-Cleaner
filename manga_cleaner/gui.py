"""
Main GUI window for Manga Cleaner.

Provides :class:`CleanerApp`, a ``tk.Tk`` application that exposes all
configuration options and runs the cleaner on a background thread.
"""

import logging
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from manga_cleaner.config import config
from manga_cleaner.folder_discovery import find_image_folders
from manga_cleaner.logging_setup import LOG_FILE, TextHandler, log
from manga_cleaner.preview_dialog import PreviewDialog
from manga_cleaner.processor import process_chapter
from manga_cleaner.tooltip import ToolTip


class CleanerApp(tk.Tk):
    """Top-level GUI window for Manga Cleaner."""

    def __init__(self):
        super().__init__()
        self.title("Manga / Image Folder Cleaner")
        self.resizable(True, True)
        self.minsize(700, 520)
        self._build_ui()
        self._attach_log_handler()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        PAD = 10

        # ── Top: folder list ──────────────────────────────────────────
        folder_frame = ttk.LabelFrame(
            self,
            text="Folders to scan (recursively searches up to 4 levels deep for folders with images)",
        )
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
        ttk.Button(btn_row, text="➕  Add folder(s)",
                   command=self._add_folders).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="➖  Remove selected",
                   command=self._remove_selected).pack(side="left")

        # ── Middle: options ───────────────────────────────────────────
        opt_frame = ttk.LabelFrame(self, text="Options")
        opt_frame.pack(fill="x", padx=PAD, pady=(6, 0))

        self.dry_run_var = tk.BooleanVar(value=config.dry_run)
        dry_cb = ttk.Checkbutton(
            opt_frame,
            text="Dry-run mode  (preview only — no files will be deleted)",
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
            spin = ttk.Spinbox(parent, from_=lo, to=hi,
                               textvariable=var, width=5)
            spin.pack(side="left", padx=(2, 12))
            if tooltip:
                ToolTip(lbl, tooltip)
                ToolTip(spin, tooltip)
            return var

        self.white_var = _labeled_spin(
            thresh_row, "White ≥", config.white_threshold, 200, 255,
            "Pixel brightness (0–255) at or above which a pixel counts as 'white'. "
            "A page is flagged as blank-white when ≥ Solid Ratio of its pixels "
            "meet this threshold. Lower = more aggressive detection.")
        self.black_var = _labeled_spin(
            thresh_row, "Black ≤", config.black_threshold, 0, 50,
            "Pixel brightness (0–255) at or below which a pixel counts as 'black'. "
            "A page is flagged as blank-black when ≥ Solid Ratio of its pixels "
            "meet this threshold. Higher = more aggressive detection.")
        self.hash_var = _labeled_spin(
            thresh_row, "Hash dist ≤", config.hash_threshold, 0, 64,
            "Maximum perceptual hash (pHash) distance between two images to "
            "consider them near-duplicates. 0 = only identical hashes match. "
            "8–10 is a good balance for manga. Higher = catches more duplicates "
            "but risks false positives.")

        solid_row = tk.Frame(opt_frame)
        solid_row.grid(row=3, column=0, sticky="w", padx=8, pady=(0, 8))
        solid_lbl = tk.Label(solid_row, text="Solid ratio ≥")
        solid_lbl.pack(side="left")
        self.solid_var = tk.DoubleVar(value=config.solid_ratio)
        solid_spin = ttk.Spinbox(
            solid_row, from_=0.5, to=1.0, increment=0.01,
            textvariable=self.solid_var, width=6, format="%.2f")
        solid_spin.pack(side="left", padx=(2, 0))
        solid_tip = (
            "Fraction of pixels that must be solid white or solid black "
            "to flag a page as blank. 0.99 means 99% of the page must "
            "be a single solid colour. Lower = more aggressive, but "
            "may flag pages with a small amount of content.")
        ToolTip(solid_lbl, solid_tip)
        ToolTip(solid_spin, solid_tip)

        # ── Advanced detection ────────────────────────────────────────
        adv_frame = ttk.LabelFrame(
            self,
            text="Advanced Detection  (optional — may produce false positives, review with dry-run first)",
        )
        adv_frame.pack(fill="x", padx=PAD, pady=(6, 0))

        def _adv_row(parent, row, label, bool_var, widgets_fn=None, tooltip=None):
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
        self.aspect_min_var = tk.DoubleVar(value=config.aspect_ratio_min)
        self.aspect_max_var = tk.DoubleVar(value=config.aspect_ratio_max)

        def _aspect_widgets(f):
            _spin_pair(f, "Min ratio", self.aspect_min_var, 0.1, 1.0, 0.01, "%.2f",
                       tooltip="Minimum acceptable width/height ratio. "
                               "Pages narrower than this are flagged. "
                               "Typical manga portrait page = 0.55-0.80.")
            _spin_pair(f, "Max ratio", self.aspect_max_var, 0.1, 2.0, 0.01, "%.2f",
                       tooltip="Maximum acceptable width/height ratio. "
                               "Pages wider than this (landscape, square) are flagged. "
                               "Typical manga portrait page = 0.55-0.80.")

        _adv_row(adv_frame, 0,
                 "🔲  Aspect ratio filter  (flag landscape / square pages)",
                 self.use_aspect_var, _aspect_widgets,
                 tooltip="Flag images whose width/height ratio falls outside the "
                         "expected manga portrait range. Catches landscape fan-art, "
                         "square ad images, and other non-standard page dimensions. "
                         "Adjust Min/Max ratio to match your manga's typical proportions.")

        # Saturation
        self.use_sat_var = tk.BooleanVar(value=False)
        self.sat_thresh_var = tk.DoubleVar(value=config.saturation_threshold)

        def _sat_widgets(f):
            _spin_pair(f, "Saturation threshold", self.sat_thresh_var,
                       0.0, 1.0, 0.01, "%.2f",
                       tooltip="Average HSV saturation (0.0-1.0) above which a page is "
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
        self.size_z_var = tk.DoubleVar(value=config.size_outlier_z)

        def _size_widgets(f):
            _spin_pair(f, "Z-score threshold", self.size_z_var,
                       1.0, 6.0, 0.1, "%.1f",
                       tooltip="How many standard deviations from the mean pixel area an "
                               "image must be to be considered an outlier. Lower = more "
                               "sensitive. 2.5 catches obviously different resolutions "
                               "without flagging minor variations.")

        _adv_row(adv_frame, 2,
                 "📐  Size outlier filter  (flag pages with very different resolution)",
                 self.use_size_var, _size_widgets,
                 tooltip="Flags images whose pixel area (width x height) is a statistical "
                         "outlier compared to other pages in the same folder. "
                         "Catches foreign inserts scanned at a completely different "
                         "resolution. Requires at least 4 images per folder to work.")

        # Text-only
        self.use_text_var = tk.BooleanVar(value=False)

        def _text_widgets(f):
            tk.Label(
                f, text="(uses CLIP ML model - requires: pip install torch transformers)",
                font=("Segoe UI", 8), fg="#888",
            ).pack(side="left")

        _adv_row(adv_frame, 3,
                 "📄  Text-only filter  (flag surveys, TOC, afterword pages)",
                 self.use_text_var, _text_widgets,
                 tooltip="Uses the CLIP machine learning model to detect pages that "
                         "contain only text and no artwork - such as table of contents, "
                         "survey forms, author notes, afterword, or credits pages. "
                         "Requires PyTorch and Transformers to be installed. "
                         "The model is downloaded on first use (~350 MB).")

        # ML (CLIP)
        self.use_ml_var = tk.BooleanVar(value=False)
        self.ml_conf_var = tk.DoubleVar(value=config.ml_confidence_threshold)

        def _ml_widgets(f):
            _spin_pair(f, "Confidence >=", self.ml_conf_var,
                       0.5, 1.0, 0.01, "%.2f",
                       tooltip="Minimum confidence score (0.0-1.0) that an image is NOT "
                               "a manga page before flagging it. Higher = fewer false "
                               "positives but may miss some non-manga content. "
                               "0.75-0.85 is a good range.")
            tk.Label(
                f, text="(requires: pip install torch transformers)",
                font=("Segoe UI", 8), fg="#888",
            ).pack(side="left")

        _adv_row(adv_frame, 4,
                 "🤖  ML classifier  (CLIP zero-shot - detects fan-art / ads)",
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
        self.run_btn = ttk.Button(
            run_row, text="▶  Run Cleaner",
            command=self._run, style="Accent.TButton")
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(
            run_row, text="⏹  Stop", command=self._stop)
        self.status_label = tk.Label(
            run_row, text="", fg="#555", font=("Segoe UI", 9))
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
        self.log_text.tag_config("header", foreground="#2980b9",
                                 font=("Consolas", 9, "bold"))
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
        handler = TextHandler(self._append_log)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)

    def _append_log(self, message: str):
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda: self._append_log(message))
            return

        self.log_text.config(state="normal")
        tag = ""
        lower = message.lower()
        if "[deleted]" in lower:
            tag = "deleted"
        elif "\u2705" in message or "success" in lower:
            tag = "ok"
        elif "[warn]" in lower or "[error]" in lower:
            tag = "warn"
        elif message.startswith("\U0001f4c1"):
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
            messagebox.showwarning("No folders",
                                   "Please add at least one folder.")
            return

        # Apply UI values to the shared config
        config.white_threshold = self.white_var.get()
        config.black_threshold = self.black_var.get()
        config.solid_ratio = self.solid_var.get()
        config.hash_threshold = self.hash_var.get()
        config.aspect_ratio_min = self.aspect_min_var.get()
        config.aspect_ratio_max = self.aspect_max_var.get()
        config.saturation_threshold = self.sat_thresh_var.get()
        config.size_outlier_z = self.size_z_var.get()
        config.ml_confidence_threshold = self.ml_conf_var.get()

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
        self.status_label.config(text="Running...")
        self.update_idletasks()

        worker = threading.Thread(target=self._run_worker, daemon=True)
        worker.start()

    def _stop(self):
        """Signal the worker thread to stop after the current chapter."""
        self._stop_event.set()
        self.stop_btn.config(state="disabled")
        self.status_label.config(
            text="Stopping... (will finish current folder)")

    # ------------------------------------------------------------------
    # Thread-safe GUI helpers
    # ------------------------------------------------------------------

    def _show_preview_from_worker(self, root_name, root_pending, root_all_images):
        """Open the PreviewDialog on the main thread, block the worker."""
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
        """Show a confirmation dialog on the main thread, block the worker."""
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
        """Heavy processing - runs on a background thread."""
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
        if use_aspect:  active_filters.append("aspect")
        if use_sat:     active_filters.append("saturation")
        if use_size:    active_filters.append("size-outlier")
        if use_text:    active_filters.append("text-only")
        if use_ml:      active_filters.append("ML/CLIP")
        filter_label = ("  |  filters: " + ", ".join(active_filters)
                        if active_filters else "")

        mode_label = "DRY RUN" if dry_run else "LIVE MODE"
        self._append_log("=" * 60)
        self._append_log(
            f"  {mode_label}  |  white>={config.white_threshold}  "
            f"black<={config.black_threshold}  solid>={config.solid_ratio}  "
            f"hash<={config.hash_threshold}{filter_label}"
        )
        self._append_log("=" * 60)

        try:
            for root_str in folders:
                if self._stop_event.is_set():
                    self._append_log("\n⏹ Processing stopped by user.")
                    break

                root = Path(root_str)
                if not root.is_dir():
                    self._append_log(
                        f"[WARN] Not a valid directory, skipping: {root_str}")
                    continue

                image_dirs = find_image_folders(root)
                self._append_log(
                    f"\n📂 Root: {root}  ({len(image_dirs)} folder(s) with images)")

                if not image_dirs:
                    self._append_log("  (no folders with images found)")
                    continue

                root_pending: dict[Path, str] = {}
                root_all_images: list[Path] = []

                for chapter in image_dirs:
                    if self._stop_event.is_set():
                        self._append_log("\n⏹ Processing stopped by user.")
                        break

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

                    if use_preview:
                        pending = stats.pop("pending_delete", {})
                        all_imgs = stats.pop("all_images", [])
                        root_pending.update(pending)
                        root_all_images.extend(all_imgs)

                    for k in totals:
                        totals[k] += stats.get(k, 0)
                    total_chapters += 1

                # Batch preview for this root
                if use_preview and root_pending and not self._stop_event.is_set():
                    self._append_log(
                        f"\n🖼  Opening preview for {len(root_pending)} "
                        f"flagged image(s) from '{root.name}'...")

                    confirmed = self._show_preview_from_worker(
                        root.name, root_pending, root_all_images)

                    skipped_count = len(root_pending) - len(confirmed)
                    totals["skipped"] += skipped_count

                    if not confirmed:
                        self._append_log(
                            f"  ⏭️ All files kept by user for '{root.name}'.\n")
                    else:
                        for p, reason in confirmed.items():
                            try:
                                p.unlink()
                                log.info(f"  [DELETED] {p.name}  ({reason})")
                                totals["removed"] += 1
                            except Exception as exc:
                                log.warning(
                                    f"  [ERROR] Could not delete {p.name}: {exc}")
                                totals["skipped"] += 1
                        self._append_log(
                            f"  ✅ '{root.name}' - deleted {len(confirmed)}, "
                            f"kept {skipped_count}\n")

            # Summary
            total_flagged = sum(
                totals[k] for k in (
                    "blank", "duplicate", "aspect", "saturation",
                    "size_outlier", "text_only", "ml_foreign",
                )
            )
            self._append_log("\n" + "=" * 60)
            self._append_log("  SUMMARY")
            self._append_log("=" * 60)
            self._append_log(f"  Folders processed     : {total_chapters}")
            self._append_log(f"  Images scanned        : {totals['scanned']}")
            self._append_log(f"  Blank pages           : {totals['blank']}")
            self._append_log(f"  Duplicates            : {totals['duplicate']}")
            if use_aspect:
                self._append_log(
                    f"  Wrong aspect ratio    : {totals['aspect']}")
            if use_sat:
                self._append_log(
                    f"  High saturation       : {totals['saturation']}")
            if use_size:
                self._append_log(
                    f"  Size outliers         : {totals['size_outlier']}")
            if use_text:
                self._append_log(
                    f"  Text-only pages       : {totals['text_only']}")
            if use_ml:
                self._append_log(
                    f"  ML non-manga          : {totals['ml_foreign']}")
            if dry_run:
                self._append_log(
                    f"  Would be deleted      : {total_flagged}")
            else:
                self._append_log(
                    f"  Deleted               : {totals['removed']}")
                self._append_log(
                    f"  Skipped / errors      : {totals['skipped']}")
            self._append_log(f"\n  Full log saved to: {LOG_FILE}")
            self._append_log("=" * 60)

        except Exception as exc:
            self._append_log(f"[ERROR] Unexpected error: {exc}")

        self.after(0, self._run_finished)

    def _run_finished(self):
        """Called on the main thread when the worker thread is done."""
        self.run_btn.config(state="normal")
        self.stop_btn.pack_forget()
        self.stop_btn.config(state="normal")
        stopped = self._stop_event.is_set()
        self.status_label.config(
            text="Stopped ⏹" if stopped else "Done ✅")


