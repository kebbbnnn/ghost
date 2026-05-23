"""Change detection via SSIM comparison.

Compares consecutive frames using Structural Similarity Index.
Maintains a ring buffer to enforce persistence before triggering.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity  # type: ignore[import-untyped]

from agent.config import SSIM_NO_CHANGE_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Outcome of a single frame-vs-previous comparison."""

    changed: bool
    ssim: float
    changed_ratio: float


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ChangeDetector:
    """SSIM-based change gate with a persistence ring buffer.

    A change is only considered *confirmed* when every entry in the ring
    buffer (of size ``persistence_frames``) reports ``changed=True``.
    This avoids spurious triggers from transient artefacts such as cursor
    blink or brief tooltip flicker.

    Parameters
    ----------
    ssim_threshold:
        Two frames with SSIM **below** this value are considered *changed*.
        Defaults to :pydata:`SSIM_NO_CHANGE_THRESHOLD`.
    persistence_frames:
        Number of consecutive changed frames required before
        :pymeth:`should_proceed` returns ``True``.
    """

    def __init__(
        self,
        ssim_threshold: float = SSIM_NO_CHANGE_THRESHOLD,
        persistence_frames: int = 3,
    ) -> None:
        self._ssim_threshold = ssim_threshold
        self._persistence_frames = persistence_frames

        self._prev_frame: np.ndarray | None = None
        self._ring_buffer: collections.deque[DetectionResult] = collections.deque(
            maxlen=persistence_frames,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: Image.Image) -> DetectionResult:
        """Compare *frame* against the previous frame.

        Parameters
        ----------
        frame:
            A PIL ``Image`` (any mode) captured from the screen.

        Returns
        -------
        DetectionResult
            Whether the frame changed, along with the raw SSIM score and
            the ratio of pixels that exceeded the diff threshold.
        """
        current_gray = self._to_grayscale(frame)

        # First frame — nothing to compare against.
        if self._prev_frame is None:
            self._prev_frame = current_gray
            result = DetectionResult(changed=False, ssim=1.0, changed_ratio=0.0)
            self._ring_buffer.append(result)
            return result

        prev = self._prev_frame

        # Resize to match dimensions if the screen resolution changed
        # between captures (e.g. Retina scaling toggle).
        if prev.shape != current_gray.shape:
            logger.debug(
                "Frame size changed from %s to %s — resizing previous frame",
                prev.shape,
                current_gray.shape,
            )
            prev_image = Image.fromarray(prev)
            prev_image = prev_image.resize(
                (current_gray.shape[1], current_gray.shape[0]),
                Image.LANCZOS,
            )
            prev = np.asarray(prev_image, dtype=np.uint8)

        # --- SSIM ----------------------------------------------------------
        ssim_value: float = structural_similarity(
            prev,
            current_gray,
            data_range=255,
        )

        # --- Pixel-level diff ---------------------------------------------
        diff = np.abs(prev.astype(np.int16) - current_gray.astype(np.int16))
        changed_pixels = int(np.count_nonzero(diff > 25))
        total_pixels = current_gray.shape[0] * current_gray.shape[1]
        changed_ratio = changed_pixels / total_pixels if total_pixels else 0.0

        # --- Store & decide ------------------------------------------------
        self._prev_frame = current_gray
        changed = ssim_value < self._ssim_threshold

        result = DetectionResult(
            changed=changed,
            ssim=round(ssim_value, 6),
            changed_ratio=round(changed_ratio, 6),
        )
        self._ring_buffer.append(result)

        logger.debug(
            "SSIM=%.4f  changed_ratio=%.4f%%  changed=%s  buffer=%d/%d",
            ssim_value,
            changed_ratio * 100,
            changed,
            sum(1 for r in self._ring_buffer if r.changed),
            self._persistence_frames,
        )
        return result

    def should_proceed(self) -> bool:
        """Return ``True`` only when the ring buffer is full **and** every
        entry reports ``changed=True``."""
        if len(self._ring_buffer) < self._persistence_frames:
            return False
        return all(r.changed for r in self._ring_buffer)

    def reset(self) -> None:
        """Clear the ring buffer and discard the stored previous frame."""
        self._ring_buffer.clear()
        self._prev_frame = None
        logger.debug("ChangeDetector reset")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_grayscale(image: Image.Image) -> np.ndarray:
        """Convert a PIL Image to a grayscale ``uint8`` numpy array."""
        return np.asarray(image.convert("L"), dtype=np.uint8)
