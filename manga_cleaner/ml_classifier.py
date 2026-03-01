"""
CLIP-based ML classifiers for Manga Cleaner.

Uses OpenAI's CLIP model (via Hugging Face ``transformers``) for zero-shot
image classification to detect non-manga content and text-only pages.

Requires ``torch`` and ``transformers`` — if unavailable the functions
gracefully return *not flagged* and log a warning.
"""

import threading
from functools import lru_cache

from PIL import Image

from manga_cleaner.config import config
from manga_cleaner.logging_setup import log

# Thread lock for safe access to the CLIP model during parallel inference
_clip_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_clip_model():
    """Load the CLIP model and processor exactly once, then cache them.

    The first call takes ~10–30 s (download + load); subsequent calls are
    instant.
    """
    from transformers import CLIPProcessor, CLIPModel

    model_name = "openai/clip-vit-base-patch32"
    log.info("  [ML] Loading CLIP model (first run may take a moment)…")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    log.info("  [ML] CLIP model ready.")
    return model, processor


# ---------------------------------------------------------------------------
# Non-manga detector
# ---------------------------------------------------------------------------

def is_non_manga_ml(img: Image.Image) -> tuple[bool, float]:
    """Use CLIP zero-shot classification to detect non-manga content.

    Returns ``(is_foreign, fanart_confidence)``.
    """
    try:
        import torch

        with _clip_lock:
            model, processor = _get_clip_model()
            inputs = processor(
                text=[config.ml_manga_label, config.ml_fanart_label],
                images=img,
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]

        fanart_prob = probs[1].item()
        return fanart_prob >= config.ml_confidence_threshold, fanart_prob

    except ImportError:
        log.warning("  [WARN] torch/transformers not installed — ML check skipped.")
        return False, 0.0
    except Exception as exc:
        log.warning(f"  [WARN] CLIP classification error: {exc}")
        return False, 0.0


# ---------------------------------------------------------------------------
# Text-only page detector
# ---------------------------------------------------------------------------

def is_text_only_page(img: Image.Image) -> tuple[bool, float]:
    """Detect pages that contain only text and no artwork using CLIP.

    Returns ``(is_text_only, confidence)``.
    """
    try:
        import torch

        with _clip_lock:
            model, processor = _get_clip_model()

            text_label = (
                "a mostly white page with only Japanese or English text, such as "
                "table of contents, survey form, author notes, afterword, or "
                "credits page with no artwork"
            )
            manga_label = (
                "a black and white Japanese manga comic book page with sequential "
                "art panels, character illustrations, speech bubbles, and visual "
                "storytelling"
            )

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
        return text_prob >= config.ml_confidence_threshold, text_prob

    except ImportError:
        log.warning("  [WARN] torch/transformers not installed — text-only ML check skipped.")
        return False, 0.0
    except Exception as exc:
        log.warning(f"  [WARN] CLIP text-only classification error: {exc}")
        return False, 0.0

