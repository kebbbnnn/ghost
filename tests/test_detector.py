"""Unit tests for the change detector module.

Tests SSIM-based change detection and persistence ring buffer logic
using real programmatically-generated PIL Images.
"""

import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agent.detector import ChangeDetector, DetectionResult
from agent.config import SSIM_NO_CHANGE_THRESHOLD, PERSISTENCE_FRAMES


def make_desktop(width: int = 800, height: int = 600, bg_color=(50, 50, 60)) -> Image.Image:
    """Generate a synthetic desktop screenshot."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    # Draw a fake "window" (white rectangle with title bar)
    draw.rectangle([100, 80, 700, 500], fill=(255, 255, 255), outline=(180, 180, 180))
    draw.rectangle([100, 80, 700, 110], fill=(230, 230, 230))  # title bar
    draw.text((110, 85), "My Application", fill=(30, 30, 30))
    return img


def make_desktop_with_text(text: str, text_pos: tuple = (150, 200)) -> Image.Image:
    """Generate a desktop with custom text in the 'window'."""
    img = make_desktop()
    draw = ImageDraw.Draw(img)
    draw.text(text_pos, text, fill=(0, 0, 0))
    return img


def make_desktop_with_cursor(x: int, y: int, visible: bool = True) -> Image.Image:
    """Generate a desktop with a cursor blinking effect."""
    img = make_desktop_with_text("Hello world")
    if visible:
        draw = ImageDraw.Draw(img)
        # Small cursor line (2px wide, 16px tall)
        draw.rectangle([x, y, x + 2, y + 16], fill=(0, 0, 0))
    return img


def make_desktop_with_error_modal() -> Image.Image:
    """Generate a desktop with an error modal overlay."""
    img = make_desktop_with_text("Some content here")
    draw = ImageDraw.Draw(img)
    # Dark overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 128))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    # Modal box
    draw.rectangle([250, 200, 550, 350], fill=(255, 255, 255), outline=(200, 50, 50), width=2)
    draw.text((280, 220), "Error", fill=(200, 50, 50))
    draw.text((280, 250), "Failed to execute prompt", fill=(80, 80, 80))
    # Retry button
    draw.rectangle([380, 300, 480, 330], fill=(66, 133, 244), outline=(50, 100, 200))
    draw.text((400, 305), "Retry", fill=(255, 255, 255))
    return img


class TestChangeDetector(unittest.TestCase):
    """Test the SSIM-based change detector."""

    def test_identical_frames_no_change(self):
        """Two identical frames should report no change."""
        detector = ChangeDetector()
        frame = make_desktop()

        # First frame — always reports no change (no comparison baseline)
        result = detector.detect(frame)
        self.assertFalse(result.changed)
        self.assertEqual(result.ssim, 1.0)

        # Second identical frame
        result = detector.detect(frame.copy())
        self.assertFalse(result.changed)
        self.assertGreater(result.ssim, SSIM_NO_CHANGE_THRESHOLD)

    def test_significant_change_detected(self):
        """A major screen change (error modal) should be detected."""
        detector = ChangeDetector()
        frame_normal = make_desktop_with_text("Hello world")
        frame_error = make_desktop_with_error_modal()

        detector.detect(frame_normal)
        result = detector.detect(frame_error)

        self.assertTrue(result.changed)
        self.assertLess(result.ssim, SSIM_NO_CHANGE_THRESHOLD)
        self.assertGreater(result.changed_ratio, 0.01)  # >1% changed

    def test_cursor_blink_small_change(self):
        """A cursor blink should produce a very small change."""
        detector = ChangeDetector()
        frame_cursor_on = make_desktop_with_cursor(250, 200, visible=True)
        frame_cursor_off = make_desktop_with_cursor(250, 200, visible=False)

        detector.detect(frame_cursor_on)
        result = detector.detect(frame_cursor_off)

        # The SSIM should be very high (cursor is tiny)
        # Whether this counts as "changed" depends on threshold
        self.assertGreater(result.ssim, 0.99)

    def test_persistence_not_met(self):
        """should_proceed should be False when ring buffer is not full."""
        detector = ChangeDetector(persistence_frames=3)
        frame1 = make_desktop()
        frame2 = make_desktop_with_error_modal()

        detector.detect(frame1)
        detector.detect(frame2)

        # Only 1 changed frame in buffer, need 3
        self.assertFalse(detector.should_proceed())

    def test_persistence_met(self):
        """should_proceed should be True after N consecutive changed frames."""
        detector = ChangeDetector(persistence_frames=3)

        # Each frame must be dramatically different from the previous one
        # so SSIM drops below the 0.995 threshold.
        frame_1 = make_desktop()                         # gray desktop
        frame_2 = make_desktop_with_error_modal()         # overlay + modal
        frame_3 = make_desktop()                          # back to gray
        frame_4 = make_desktop_with_error_modal()         # modal again

        # Baseline
        detector.detect(frame_1)

        # Feed 3 large changes (each flip is dramatic enough)
        detector.detect(frame_2)  # gray → modal
        detector.detect(frame_3)  # modal → gray
        detector.detect(frame_4)  # gray → modal

        self.assertTrue(detector.should_proceed())

    def test_persistence_broken_by_no_change(self):
        """An unchanged frame in the middle should break persistence."""
        detector = ChangeDetector(persistence_frames=3)
        frame_normal = make_desktop()
        frame_changed = make_desktop_with_error_modal()

        detector.detect(frame_normal)
        detector.detect(frame_changed)  # changed
        detector.detect(frame_normal)   # back to normal (not changed relative to changed)

        # Persistence broken
        self.assertFalse(detector.should_proceed())

    def test_reset_clears_state(self):
        """reset() should clear buffer and previous frame."""
        detector = ChangeDetector()
        frame = make_desktop()

        detector.detect(frame)
        detector.detect(make_desktop_with_error_modal())
        detector.reset()

        # After reset, first frame should return no-change baseline
        result = detector.detect(frame)
        self.assertFalse(result.changed)
        self.assertEqual(result.ssim, 1.0)


class TestSyntheticFrameGeneration(unittest.TestCase):
    """Sanity checks on the test frame generators."""

    def test_desktop_is_rgb(self):
        img = make_desktop()
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.size, (800, 600))

    def test_error_modal_differs_from_normal(self):
        normal = make_desktop_with_text("Hello")
        error = make_desktop_with_error_modal()
        # Convert to numpy and check they're actually different
        diff = np.abs(
            np.array(normal, dtype=np.float32) - np.array(error, dtype=np.float32)
        )
        self.assertGreater(diff.mean(), 5.0)  # Meaningful pixel difference


if __name__ == "__main__":
    unittest.main()
