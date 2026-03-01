"""
Image preview dialog for Manga Cleaner.

Modal dialog that shows each flagged image one-by-one and lets the user
decide whether to delete or keep it, with optional context thumbnails of
neighbouring pages.
"""

from pathlib import Path

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class PreviewDialog(tk.Toplevel):
    """Review flagged images before deletion.

    Parameters
    ----------
    parent : tk.Tk
        Parent tkinter window.
    root_name : str
        Display name (e.g. manga root folder name).
    to_delete : dict[Path, str]
        ``{path: reason}`` of all flagged files across chapters.
    all_images : list[Path]
        Flat sorted list of **all** image paths across chapters (used
        to find neighbours for context thumbnails).

    Keyboard Shortcuts
    ------------------
    ``D`` / ``Delete``   — mark current image for deletion
    ``K`` / ``Right``    — keep current image (skip)
    ``B`` / ``Left``     — go back and undo previous decision
    ``A``                — mark ALL remaining images for deletion
    ``N`` / ``Escape``   — keep ALL remaining images and close
    ``C``                — toggle context thumbnails on/off

    After the dialog closes, call :meth:`get_confirmed` to retrieve the
    dict of ``{Path: reason}`` for files the user confirmed for deletion.
    """

    MAX_PREVIEW_SIZE = 650
    CONTEXT_THUMB_SIZE = 180

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
        self._history: list[str] = []

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

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _load_thumbnail(path: Path, max_size: int):
        """Load an image and return a scaled ``PhotoImage``, or ``None``."""
        try:
            img = Image.open(path)
            img.load()
            img = img.convert("RGB")
            w, h = img.size
            if w > max_size or h > max_size:
                ratio = min(max_size / w, max_size / h)
                img = img.resize(
                    (int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS
                )
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    # ---- UI ----------------------------------------------------------------

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
            self, text="", font=("Segoe UI", 9), fg="#c0392b",
            wraplength=880, justify="left")
        self.reason_label.pack(fill="x", padx=10, pady=(0, 4))

        # ── Image area: [prev thumb] [main image] [next thumb] ──
        img_area = tk.Frame(self)
        img_area.pack(fill="both", expand=True, padx=10, pady=4)

        # Previous context thumbnail
        self._prev_frame = tk.Frame(img_area, width=self.CONTEXT_THUMB_SIZE + 10)
        self._prev_frame.pack(side="left", fill="y", padx=(0, 6))
        self._prev_frame.pack_propagate(False)

        tk.Label(self._prev_frame, text="← prev",
                 font=("Segoe UI", 7), fg="#888").pack(side="top")
        self._prev_label = tk.Label(self._prev_frame, bg="#333", anchor="center")
        self._prev_label.pack(fill="both", expand=True)

        # Next context thumbnail
        self._next_frame = tk.Frame(img_area, width=self.CONTEXT_THUMB_SIZE + 10)
        self._next_frame.pack(side="right", fill="y", padx=(6, 0))
        self._next_frame.pack_propagate(False)

        tk.Label(self._next_frame, text="next →",
                 font=("Segoe UI", 7), fg="#888").pack(side="top")
        self._next_label = tk.Label(self._next_frame, bg="#333", anchor="center")
        self._next_label.pack(fill="both", expand=True)

        # Main image
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
            btn_frame, text="🗑  Delete  (D)", command=self._on_delete)
        self.delete_btn.pack(side="left", padx=(0, 6))

        self.keep_btn = ttk.Button(
            btn_frame, text="✅  Keep  (K)", command=self._on_keep)
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
            text=(
                "Shortcuts:  D / Del → delete  |  K / → → keep  "
                "|  B / ← → back  |  A → delete all  "
                "|  N / Esc → keep all  |  C → toggle context"
            ),
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

    # ---- Navigation --------------------------------------------------------

    def _get_neighbour(self, path: Path, offset: int):
        """Return the image *offset* positions away in the all-images list."""
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

        # Main preview
        self._photo_main = self._load_thumbnail(path, self.MAX_PREVIEW_SIZE)
        if self._photo_main:
            self.image_label.config(image=self._photo_main, text="")
        else:
            self.image_label.config(
                image="", text="⚠ Could not load preview",
                fg="white", font=("Segoe UI", 12))

        # Context thumbnails
        prev_path = self._get_neighbour(path, -1)
        next_path = self._get_neighbour(path, +1)

        if prev_path:
            self._photo_prev = self._load_thumbnail(
                prev_path, self.CONTEXT_THUMB_SIZE)
            if self._photo_prev:
                self._prev_label.config(image=self._photo_prev, text="")
            else:
                self._prev_label.config(image="", text="—", fg="#888")
        else:
            self._photo_prev = None
            self._prev_label.config(image="", text="(start)", fg="#888")

        if next_path:
            self._photo_next = self._load_thumbnail(
                next_path, self.CONTEXT_THUMB_SIZE)
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

    # ---- Actions -----------------------------------------------------------

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
        self.destroy()

    def get_confirmed(self) -> dict:
        """Return ``{Path: reason}`` for files the user confirmed."""
        return self._confirmed

