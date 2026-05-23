"""Frame capture providers.

Defines the FrameProvider protocol and implementations:
- WindowFrameProvider: captures a specific app's window using mss
- ImageListFrameProvider: replays a list of PIL Images (for testing)
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import mss
import mss.tools
from PIL import Image

from agent.accessibility import WindowInfo, find_window

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class FrameProvider(Protocol):
    """Produces frames (screenshots) on demand."""

    def capture(self) -> Image.Image | None:
        """Capture a single frame.

        Returns:
            A PIL Image of the captured region, or *None* if no frame
            could be produced (window disappeared, list exhausted, etc.).
        """
        ...


# ---------------------------------------------------------------------------
# Window-scoped provider (mss)
# ---------------------------------------------------------------------------
class WindowFrameProvider:
    """Captures screenshots of a specific application window via ``mss``.

    On each :meth:`capture` call the window bounds are refreshed from the
    accessibility tree so that moved/resized windows are tracked
    automatically.

    Note on Retina displays:
        ``mss`` captures at the **native** (physical-pixel) resolution.
        A logical 800×600 window on a 2× Retina display will produce a
        1600×1200 image.  The :attr:`last_captured_size` attribute stores
        the *actual* pixel dimensions of the most recent capture for
        coordinate mapping.
    """

    def __init__(self, app_name: str) -> None:
        self.app_name = app_name

        # Attempt an initial window lookup so the caller gets immediate
        # feedback if the app isn't running / isn't accessible.
        self._window_info: WindowInfo | None = find_window(app_name)
        if self._window_info is None:
            logger.warning(
                "WindowFrameProvider: could not find window for '%s' at init. "
                "capture() will keep retrying.",
                app_name,
            )
        else:
            logger.info(
                "WindowFrameProvider initialised for '%s' — "
                "window '%s' at (%d, %d) %dx%d",
                app_name,
                self._window_info.window_title,
                self._window_info.x,
                self._window_info.y,
                self._window_info.width,
                self._window_info.height,
            )

        # Stores the (width, height) in actual captured pixels (may differ
        # from logical window size on Retina displays).
        self.last_captured_size: tuple[int, int] | None = None

    def capture(self) -> Image.Image | None:
        """Grab the window region and return it as an RGBA PIL Image.

        The window bounds are refreshed every call so moves/resizes are
        tracked.  Returns *None* if the window cannot be located.
        """
        # Refresh window bounds — the user may have moved or resized.
        self._window_info = find_window(self.app_name)
        if self._window_info is None:
            logger.warning(
                "WindowFrameProvider: window for '%s' not found — skipping frame",
                self.app_name,
            )
            self.last_captured_size = None
            return None

        wi = self._window_info
        monitor = {
            "left": wi.x,
            "top": wi.y,
            "width": wi.width,
            "height": wi.height,
        }

        try:
            with mss.mss() as sct:
                raw = sct.grab(monitor)
                # mss returns BGRA; convert to PIL Image (will be BGRA).
                image = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

            self.last_captured_size = image.size
            logger.debug(
                "Captured %dx%d frame for '%s' (logical window %dx%d)",
                image.width,
                image.height,
                self.app_name,
                wi.width,
                wi.height,
            )
            return image

        except Exception as exc:
            logger.error(
                "mss capture failed for '%s': %s",
                self.app_name,
                exc,
                exc_info=True,
            )
            self.last_captured_size = None
            return None


# ---------------------------------------------------------------------------
# Replay provider (for testing / offline analysis)
# ---------------------------------------------------------------------------
class ImageListFrameProvider:
    """Replays a pre-loaded sequence of PIL Images.

    Useful for deterministic testing and offline replay of recorded frames.
    """

    def __init__(self, images: list[Image.Image]) -> None:
        if not images:
            logger.warning("ImageListFrameProvider created with an empty image list")
        self._images = images
        self._index = 0

    def capture(self) -> Image.Image | None:
        """Return the next image in the list, or *None* when exhausted."""
        if self._index >= len(self._images):
            logger.debug(
                "ImageListFrameProvider exhausted (%d images consumed)",
                len(self._images),
            )
            return None

        image = self._images[self._index]
        self._index += 1
        logger.debug(
            "ImageListFrameProvider: returning image %d/%d (%dx%d)",
            self._index,
            len(self._images),
            image.width,
            image.height,
        )
        return image

    @property
    def remaining(self) -> int:
        """Number of images remaining in the sequence."""
        return max(0, len(self._images) - self._index)

    def reset(self) -> None:
        """Reset the provider to replay from the beginning."""
        self._index = 0
        logger.debug("ImageListFrameProvider reset to index 0")
