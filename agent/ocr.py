"""macOS Vision framework OCR wrapper.

Uses VNRecognizeTextRequest via PyObjC to extract text from screenshots.
No external binaries required — uses the built-in macOS Vision framework.
"""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard: Vision framework is only available on macOS
# ---------------------------------------------------------------------------
_VISION_AVAILABLE = False

if platform.system() == "Darwin":
    try:
        from Vision import VNImageRequestHandler, VNRecognizeTextRequest  # type: ignore[import-untyped]
        from Quartz import (  # type: ignore[import-untyped]
            CGDataProviderCreateWithData,
            CGImageCreate,
            CGColorSpaceCreateWithName,
            kCGColorSpaceGenericRGB,
            kCGImageAlphaLast,
            kCGBitmapByteOrderDefault,
        )

        _VISION_AVAILABLE = True
    except ImportError:
        logger.warning(
            "PyObjC Vision/Quartz bindings not installed. "
            "Install pyobjc-framework-Vision and pyobjc-framework-Quartz "
            "to enable OCR."
        )
else:
    logger.warning(
        "OCR module requires macOS with the Vision framework. "
        "Current platform: %s",
        platform.system(),
    )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OCRResult:
    """A single recognised text region.

    Attributes:
        text: The recognised string.
        confidence: Recognition confidence in [0, 1].
        bbox: Normalised bounding box (x, y, width, height) with origin
              at the **bottom-left** corner, matching Vision framework
              conventions.  All values are in [0, 1].
    """

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_text(image: Image.Image) -> list[OCRResult]:
    """Run macOS Vision OCR on a PIL Image and return recognised text regions.

    Args:
        image: An arbitrary-mode PIL Image (will be converted to RGBA).

    Returns:
        A list of :class:`OCRResult` ordered by the Vision framework's
        default ordering (roughly top-to-bottom, left-to-right).

    Raises:
        RuntimeError: If the Vision framework is not available on this
            platform or the recognition request fails.
    """
    if not _VISION_AVAILABLE:
        raise RuntimeError(
            "Vision framework is not available. "
            "This function requires macOS with pyobjc-framework-Vision."
        )

    # ------------------------------------------------------------------
    # 1. Convert PIL Image → raw RGBA bytes → CGImage
    # ------------------------------------------------------------------
    rgba_image = image.convert("RGBA")
    width, height = rgba_image.size
    raw_bytes = rgba_image.tobytes()
    bytes_per_row = width * 4  # 4 channels (RGBA), 1 byte each

    logger.debug("Creating CGImage from %dx%d RGBA image (%d bytes)", width, height, len(raw_bytes))

    color_space = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGB)
    data_provider = CGDataProviderCreateWithData(None, raw_bytes, len(raw_bytes), None)

    cg_image = CGImageCreate(
        width,
        height,
        8,               # bits per component
        32,              # bits per pixel (4 × 8)
        bytes_per_row,
        color_space,
        kCGImageAlphaLast | kCGBitmapByteOrderDefault,
        data_provider,
        None,            # decode array
        False,           # should interpolate
        0,               # rendering intent (kCGRenderingIntentDefault)
    )

    if cg_image is None:
        raise RuntimeError("Failed to create CGImage from PIL image data")

    # ------------------------------------------------------------------
    # 2. Create Vision request + handler
    # ------------------------------------------------------------------
    handler = VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
    request = VNRecognizeTextRequest.alloc().init()

    # Recognition level: 1 = VNRequestTextRecognitionLevelAccurate
    request.setRecognitionLevel_(1)

    logger.debug("Executing VNRecognizeTextRequest (accurate mode)")

    success, error = handler.performRequests_error_([request], None)

    if not success:
        error_msg = str(error) if error else "unknown Vision error"
        logger.error("Vision OCR request failed: %s", error_msg)
        raise RuntimeError(f"Vision OCR request failed: {error_msg}")

    # ------------------------------------------------------------------
    # 3. Extract results
    # ------------------------------------------------------------------
    observations = request.results()
    if observations is None:
        logger.debug("Vision returned no observations")
        return []

    results: list[OCRResult] = []
    for obs in observations:
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue

        candidate = candidates[0]
        text = candidate.string()
        confidence = candidate.confidence()

        # boundingBox → CGRect (origin.x, origin.y, size.width, size.height)
        # Normalised [0,1], origin at bottom-left.
        bb = obs.boundingBox()
        bbox = (bb.origin.x, bb.origin.y, bb.size.width, bb.size.height)

        results.append(OCRResult(text=text, confidence=confidence, bbox=bbox))

    logger.info(
        "OCR extracted %d text region(s) from %dx%d image",
        len(results),
        width,
        height,
    )
    return results
