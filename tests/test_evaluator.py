"""End-to-end test for the condition evaluator.

Generates a synthetic error modal screenshot, runs real macOS OCR on it,
and sends it to the LLM evaluator to verify condition matching.

Requires:
  - Ghost proxy running on http://127.0.0.1:8100
  - macOS with Vision framework (for OCR)
"""

import asyncio
import sys
import unittest

from PIL import Image, ImageDraw, ImageFont

from agent.evaluator import ConditionEvaluator


def make_error_modal_screenshot(width: int = 800, height: int = 600) -> Image.Image:
    """Generate a synthetic screenshot with an error modal."""
    font = ImageFont.load_default(size=16)
    # Background "app window"
    img = Image.new("RGB", (width, height), (45, 45, 55))
    draw = ImageDraw.Draw(img)

    # App content area
    draw.rectangle([0, 0, width, 40], fill=(60, 60, 70))  # title bar
    draw.text((20, 12), "My Application — editor.py", fill=(200, 200, 200), font=font)
    draw.text((50, 100), "def hello_world():", fill=(150, 200, 150), font=font)
    draw.text((80, 125), 'print("Hello World")', fill=(200, 200, 150), font=font)

    # Semi-transparent overlay effect (darkened background)
    overlay = Image.new("RGB", (width, height), (20, 20, 30))
    img = Image.blend(img, overlay, 0.4)
    draw = ImageDraw.Draw(img)

    # Error modal
    modal_x, modal_y = 200, 180
    modal_w, modal_h = 400, 200
    draw.rectangle(
        [modal_x, modal_y, modal_x + modal_w, modal_y + modal_h],
        fill=(255, 255, 255),
        outline=(220, 60, 60),
        width=2,
    )

    # Modal title
    draw.text((modal_x + 20, modal_y + 15), "Error", fill=(220, 60, 60), font=font)

    # Modal message
    draw.text(
        (modal_x + 20, modal_y + 55),
        "Failed to execute prompt.",
        fill=(80, 80, 80),
        font=font,
    )
    draw.text(
        (modal_x + 20, modal_y + 80),
        "An unexpected error occurred.",
        fill=(120, 120, 120),
        font=font,
    )

    # Retry button
    btn_x, btn_y = modal_x + 250, modal_y + 140
    btn_w, btn_h = 120, 40
    draw.rectangle(
        [btn_x, btn_y, btn_x + btn_w, btn_y + btn_h],
        fill=(66, 133, 244),
    )
    draw.text((btn_x + 35, btn_y + 10), "Retry", fill=(255, 255, 255), font=font)

    # Cancel button
    cancel_x = modal_x + 110
    draw.rectangle(
        [cancel_x, btn_y, cancel_x + btn_w, btn_y + btn_h],
        fill=(230, 230, 230),
    )
    draw.text((cancel_x + 30, btn_y + 10), "Cancel", fill=(80, 80, 80), font=font)

    return img


def make_normal_screenshot(width: int = 800, height: int = 600) -> Image.Image:
    """Generate a normal screenshot with no error modal."""
    img = Image.new("RGB", (width, height), (45, 45, 55))
    draw = ImageDraw.Draw(img)

    # App content area
    draw.rectangle([0, 0, width, 40], fill=(60, 60, 70))
    draw.text((20, 12), "My Application — editor.py", fill=(200, 200, 200))
    draw.text((50, 100), "def hello_world():", fill=(150, 200, 150))
    draw.text((80, 125), 'print("Hello World")', fill=(200, 200, 150))
    draw.text((50, 165), "# Everything is running fine", fill=(130, 130, 130))

    return img


import socket

def is_proxy_running(host="127.0.0.1", port=8100):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False

@unittest.skipUnless(is_proxy_running(), "Requires running Ghost proxy on port 8100")
class TestConditionEvaluatorE2E(unittest.TestCase):
    """E2E tests for the condition evaluator (requires running Ghost proxy)."""

    def setUp(self):
        self.instructions = [
            "When an error modal appears saying there was a failure or error executing a prompt, click the Retry button",
        ]
        self.evaluator = ConditionEvaluator(instructions=self.instructions)

    def test_error_modal_triggers_condition(self):
        """An error modal screenshot should trigger the instruction."""
        screenshot = make_error_modal_screenshot()

        # Run OCR
        ocr_text = ""
        try:
            from agent.ocr import extract_text, format_ocr_for_prompt
            ocr_results = extract_text(screenshot)
            ocr_text = format_ocr_for_prompt(ocr_results, image_size=screenshot.size)
            print(f"\nOCR extracted:\n{ocr_text}")
        except Exception as e:
            print(f"\nOCR unavailable ({e}), proceeding with screenshot only")

        result = asyncio.run(self.evaluator.evaluate(
            screenshot=screenshot,
            ocr_text=ocr_text,
            ax_snapshot={"app_name": "TestApp", "window_title": "editor.py"},
        ))

        print(f"\nEval result: matched={result.condition_matched}, "
              f"reasoning={result.reasoning}")
        if result.action_json:
            print(f"Action: {result.action_json}")

        self.assertTrue(result.condition_matched, "Error modal should trigger the condition")
        self.assertEqual(result.matched_instruction_index, 0)
        self.assertIsNotNone(result.action_json)
        self.assertEqual(result.action_json.get("action"), "click")

    def test_normal_screen_no_trigger(self):
        """A normal screen without errors should NOT trigger any condition."""
        screenshot = make_normal_screenshot()

        result = asyncio.run(self.evaluator.evaluate(
            screenshot=screenshot,
            ocr_text="def hello_world():\n  print('Hello World')\n# Everything is running fine",
            ax_snapshot={"app_name": "TestApp", "window_title": "editor.py"},
        ))

        print(f"\nEval result: matched={result.condition_matched}, "
              f"reasoning={result.reasoning}")

        self.assertFalse(result.condition_matched, "Normal screen should NOT trigger")


if __name__ == "__main__":
    # Check if --e2e flag is passed, otherwise skip
    if "--e2e" not in sys.argv:
        print("Skipping E2E tests. Run with --e2e flag to execute.")
        print("  Requires Ghost proxy running on http://127.0.0.1:8100")
        print("\n  python3 tests/test_evaluator.py --e2e")
        sys.exit(0)

    sys.argv.remove("--e2e")
    unittest.main()
