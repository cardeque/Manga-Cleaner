# Manga / Image Folder Cleaner

A GUI tool that scans manga (or any image) folders **recursively** (up to 4 levels deep), detects unwanted images, and lets you review and delete them — with a full visual preview.

---

## Features

### Core Detection (always active)

| Detection | Method | What it catches |
|---|---|---|
| **Exact duplicates** | MD5 hash of raw pixel bytes | Byte-for-byte identical images, even with different filenames |
| **Near-duplicates** | Perceptual hash (`pHash`) distance | Same image re-encoded, resized, or slightly edited |
| **Blank pages** | Grayscale histogram analysis | Images where ≥99% of pixels are near-white or near-black |

### Advanced Detection (optional — enable in the GUI)

| Filter | Method | What it catches |
|---|---|---|
| 🔲 **Aspect ratio** | Width/height ratio check | Landscape, square, or oddly-shaped pages (ads, fan-art) |
| 🎨 **Saturation** | HSV colour analysis | Highly colourful images in a B&W manga (fan-art, colour inserts) |
| 📐 **Size outlier** | Z-score of pixel area | Pages scanned at a completely different resolution |
| 📄 **Text-only** | CLIP ML zero-shot classification | Table of contents, surveys, afterwords, credits pages |
| 🤖 **ML classifier** | CLIP ML zero-shot classification | Fan-art, advertisements, promotional illustrations |

> The ML filters (📄 and 🤖) require `torch` and `transformers` — see [Optional ML Dependencies](#optional-ml-dependencies) below.

---

## Getting Started

### 1. Install Python

Download and install [Python 3.10 or later](https://www.python.org/downloads/). During installation, **check "Add Python to PATH"**.

### 2. Install Dependencies

Open a terminal (Command Prompt, PowerShell, or your OS terminal) and run:

```bash
pip install Pillow imagehash
```

### 3. Run the Tool

```bash
python manga_cleaner.py
```

A GUI window will open.

---

## Step-by-Step Usage Guide

### Step 1 — Add Folders

Click **"➕ Add folder(s)"** and select the root folder of your manga library (e.g. `C:\Manga\One Piece`).

The tool will automatically search **up to 4 levels deep** for subfolders that contain images. You can add multiple root folders.

**Example folder structure:**

```
C:\Manga\One Piece\          ← add this folder
├── Volume 01\
│   ├── Chapter 001\         ← found (contains images)
│   └── Chapter 002\         ← found
├── Volume 02\
│   ├── Chapter 003\         ← found
│   └── Chapter 004\         ← found
└── Extras\                  ← found (if it has images)
```

To remove a folder from the list, select it and click **"➖ Remove selected"**.

### Step 2 — Configure Options

All options have **tooltips** — hover your mouse over any checkbox or control for a detailed explanation.

#### Basic Options

| Option | Default | Description |
|---|---|---|
| **Dry-run mode** | ✅ ON | Scans and logs results without deleting anything. **Start here.** |
| **Preview images** | ✅ ON | When dry-run is OFF, shows each flagged image for manual review before deletion |
| **White ≥** | `250` | Brightness threshold for white-blank detection (0–255) |
| **Black ≤** | `5` | Brightness threshold for black-blank detection (0–255) |
| **Hash dist ≤** | `8` | Max perceptual hash distance for near-duplicate detection (0–64, lower = stricter) |
| **Solid ratio ≥** | `0.99` | Fraction of pixels that must be solid to flag as blank (0.5–1.0) |

#### Advanced Filters

Enable these checkboxes to activate additional detection. Each has its own threshold control:

- **🔲 Aspect ratio** — Set min/max width-to-height ratio (default 0.55–0.80 for typical manga portrait pages)
- **🎨 Saturation** — Set the colour saturation threshold (default 0.18)
- **📐 Size outlier** — Set the Z-score threshold (default 2.5, needs ≥4 images per folder)
- **📄 Text-only** — Detects text-only pages using CLIP ML model (needs `torch` + `transformers`)
- **🤖 ML classifier** — Detects non-manga content using CLIP (only scans first & last 20% of each folder). Set the confidence threshold (default 0.75)

### Step 3 — Run with Dry-Run First

1. Make sure **"Dry-run mode"** is checked
2. Click **"▶ Run Cleaner"**
3. Review the log output — it will list every flagged file and the reason, without deleting anything

```
📂 Root: C:\Manga\One Piece  (24 folder(s) with images)
📁 Processing: Volume 01\Chapter 001
  🔍 Analyzing folder: Chapter 001
  [DRY RUN] Flagged for removal (2 file(s)):
    - 001_blank.jpg                           blank (white)
    - 015_copy.jpg                            exact duplicate of '015.jpg'
  ✅ Folder 'Chapter 001' analysis complete (dry run)
```

### Step 4 — Review and Delete

Once you're happy with the dry-run results:

1. **Uncheck** "Dry-run mode"
2. Make sure **"Preview images before delete"** is checked (recommended)
3. Click **"▶ Run Cleaner"**

After all folders are scanned, a **Preview Dialog** opens showing each flagged image:

```
┌─────────────────────────────────────────────┐
│  [3 / 12]  page_017.jpg     📁 Chapter 003 │
│  Reason: blank (white)                      │
│                                             │
│  ← prev  │   [main image]   │  next →      │
│           │                  │              │
│  ☑ Show neighbouring pages for context (C)  │
│                                             │
│  🗑 Delete (D)  ✅ Keep (K)  ⬅ Back (B)    │
│  🗑 Delete ALL (A)  ⏭ Keep ALL (N)         │
└─────────────────────────────────────────────┘
```

#### Preview Keyboard Shortcuts

| Key | Action |
|---|---|
| `D` or `Delete` | Mark image for deletion, go to next |
| `K` or `→` (Right) | Keep image, go to next |
| `B` or `←` (Left) | Go back and undo previous decision |
| `A` | Mark ALL remaining images for deletion |
| `N` or `Escape` | Keep ALL remaining images and close |
| `C` | Toggle context thumbnails (prev/next images) |

Only images you explicitly mark with **Delete** will be removed. Everything else is kept.

### Step 5 — Review the Log

After processing, a summary is displayed:

```
============================================================
  SUMMARY
============================================================
  Folders processed     : 24
  Images scanned        : 4512
  Blank pages           : 12
  Duplicates            : 3
  Deleted               : 10
  Skipped / errors      : 5
```

A full log is also saved to `cleaning_log.txt` in the same directory as the script.

---

## Optional ML Dependencies

The **Text-only filter** (📄) and **ML classifier** (🤖) require PyTorch and Hugging Face Transformers:

```bash
pip install torch transformers
```

- The CLIP model (`openai/clip-vit-base-patch32`, ~350 MB) is downloaded automatically on first use
- If these packages are not installed, the ML features are simply disabled — everything else works normally
- The ML classifier only scans the **first and last 20%** of each folder (where non-manga inserts typically appear)

---

## Configuration Reference

All defaults can be edited at the top of `manga_cleaner.py`, but most users should use the GUI controls instead.

| Variable | Default | GUI Control | Description |
|---|---|---|---|
| `WHITE_THRESHOLD` | `250` | White ≥ | Brightness at or above = "white" pixel |
| `BLACK_THRESHOLD` | `5` | Black ≤ | Brightness at or below = "black" pixel |
| `SOLID_RATIO` | `0.99` | Solid ratio ≥ | Fraction of solid pixels to flag as blank |
| `HASH_THRESHOLD` | `8` | Hash dist ≤ | Max pHash distance for near-duplicates |
| `DRY_RUN` | `True` | Dry-run checkbox | Start in preview-only mode |
| `ASPECT_RATIO_MIN` | `0.55` | Min ratio | Minimum acceptable width/height |
| `ASPECT_RATIO_MAX` | `0.80` | Max ratio | Maximum acceptable width/height |
| `SATURATION_THRESHOLD` | `0.18` | Saturation threshold | Max HSV saturation for B&W manga |
| `SIZE_OUTLIER_Z` | `2.5` | Z-score threshold | Z-score to flag resolution outliers |
| `ML_CONFIDENCE_THRESHOLD` | `0.75` | Confidence ≥ | Min confidence to flag as non-manga |
| `MAX_SCAN_DEPTH` | `4` | *(code only)* | Max folder recursion depth |

---

## Supported Image Formats

`.jpg` · `.jpeg` · `.png` · `.webp` · `.bmp` · `.gif` · `.tiff` · `.tif`

---

## Performance

- Image loading, hashing, blank detection, aspect ratio, and saturation checks all run in **parallel** using a thread pool (up to 8 workers)
- ML inference uses a dedicated thread pool (1–2 workers) with a thread lock to prevent GPU/memory contention
- The GUI stays responsive during processing — all heavy work runs on a background thread
- The ML classifier only checks first/last 20% of each folder (where non-manga inserts typically appear)

---

## Tips

- **Always start with dry-run ON** to review what would be deleted
- **Use the preview dialog** to visually confirm each flagged image — the context thumbnails (prev/next pages) help you spot false positives
- **Adjust thresholds** if you get too many or too few results — hover over each control for tooltip guidance
- **The Back button** (`B` or `←`) in the preview dialog lets you undo a mistake without starting over
- **Duplicate detection** keeps the file with the alphabetically lowest filename within each group
- **Log file** (`cleaning_log.txt`) contains a permanent record of all actions taken
