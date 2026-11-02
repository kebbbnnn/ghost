"""LLM-based condition evaluator.

Builds a multimodal prompt from the current screen state (screenshot,
OCR text, accessibility snapshot) and the user's instructions, sends it
to the Ghost proxy, and parses the structured JSON response to
determine if any instruction condition is met.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field

import httpx
from PIL import Image

from agent.config import PROXY_URL, PROXY_MODEL, CONTEXT_IMAGE_MAX_WIDTH

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating instructions against current screen state."""

    condition_matched: bool
    matched_instruction_index: int | None = None
    reasoning: str = ""
    action_json: dict | None = None


SYSTEM_PROMPT = """\
You are a screen observation agent. You continuously watch a target application \
window and evaluate whether any of the user's trigger conditions are currently \
met on screen.

You will receive:
1. A screenshot of the target application window.
2. OCR-extracted text visible on screen.
3. An accessibility snapshot (focused element, visible buttons, window title).
4. A numbered list of user instructions, each describing a condition and an action.

Your job:
- Examine the screenshot, OCR text, and accessibility data.
- Determine if ANY of the numbered instructions' conditions are currently met.
- If a condition is met, return the appropriate action.
- Actions use WINDOW-RELATIVE coordinates (top-left of the window is 0,0).
- Use the OCR bounding boxes and visual layout to estimate button/element positions.

You MUST respond with valid JSON in exactly this format:

If a condition IS matched:
{
  "condition_matched": true,
  "matched_instruction_index": <0-based index>,
  "reasoning": "<brief explanation of what you see and why it matches>",
  "action": {"action": "<type>", ...action params...}
}

If NO condition is matched:
{
  "condition_matched": false,
  "reasoning": "<brief explanation of current screen state>"
}

Action types and their parameters:
- {"action": "click", "x": <int>, "y": <int>}
- {"action": "double_click", "x": <int>, "y": <int>}
- {"action": "right_click", "x": <int>, "y": <int>}
- {"action": "type", "text": "<string>"}
- {"action": "hotkey", "keys": ["<key1>", "<key2>"]}
- {"action": "scroll", "x": <int>, "y": <int>, "direction": "up"|"down", "amount": <int>}
- {"action": "wait", "seconds": <float>}

IMPORTANT:
- Coordinates are relative to the APPLICATION WINDOW, not the screen.
- Be precise with coordinates — aim for the center of buttons/elements.
- Only match a condition if you are confident it is met. When uncertain, do NOT match.
- Never invent actions not described in the instructions.
"""


def _image_to_base64(image: Image.Image, max_width: int = CONTEXT_IMAGE_MAX_WIDTH) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string, downscaling if needed."""
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    image = image.convert("RGB")
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_user_content(
    screenshot: Image.Image,
    ocr_text: str,
    ax_snapshot: dict,
    instructions: list[str],
) -> list[dict]:
    """Build the multimodal user message content."""
    # Instructions block
    instructions_text = "## Instructions (evaluate each condition):\n\n"
    for i, inst in enumerate(instructions):
        instructions_text += f"{i}. {inst}\n"

    # OCR block
    ocr_block = f"## OCR Text Detected on Screen:\n\n```\n{ocr_text}\n```" if ocr_text else "## OCR Text: (none detected)"

    # Accessibility block
    ax_block = f"## Accessibility Snapshot:\n\n```json\n{json.dumps(ax_snapshot, indent=2, default=str)}\n```"

    # Screenshot as base64 image
    b64 = _image_to_base64(screenshot)

    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
        {
            "type": "text",
            "text": f"{instructions_text}\n\n{ocr_block}\n\n{ax_block}",
        },
    ]


class ConditionEvaluator:
    """Evaluates user instructions against the current screen state via an LLM."""

    def __init__(
        self,
        instructions: list[str],
        proxy_url: str = PROXY_URL,
        model: str = PROXY_MODEL,
    ):
        self.instructions = instructions
        self.proxy_url = proxy_url
        self.model = model

    async def evaluate(
        self,
        screenshot: Image.Image,
        ocr_text: str = "",
        ax_snapshot: dict | None = None,
    ) -> EvalResult:
        """Evaluate whether any instruction condition is met.

        Args:
            screenshot: Current window screenshot.
            ocr_text: Concatenated OCR text from the window.
            ax_snapshot: Accessibility snapshot dict (app_name, window_title, etc.)

        Returns:
            EvalResult with match status and optional action.
        """
        if ax_snapshot is None:
            ax_snapshot = {}

        user_content = _build_user_content(
            screenshot, ocr_text, ax_snapshot, self.instructions
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(self.proxy_url, json=payload)
                resp.raise_for_status()

            body = resp.json()
            content = body["choices"][0]["message"]["content"]

            # Parse JSON from the LLM response — handle markdown code fences
            content = content.strip()
            if content.startswith("```"):
                # Strip code fences
                lines = content.split("\n")
                # Remove first and last lines (```json and ```)
                lines = [
                    l for l in lines if not l.strip().startswith("```")
                ]
                content = "\n".join(lines).strip()

            result = None
            try:
                result = json.loads(content)
            except Exception:
                try:
                    import json_repair
                    result = json_repair.repair_json(content, return_objects=True)
                except Exception:
                    pass

            if not isinstance(result, dict):
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    cand = content[start:end + 1]
                    try:
                        import json_repair
                        result = json_repair.repair_json(cand, return_objects=True)
                    except Exception:
                        pass

            if not isinstance(result, dict):
                raise json.JSONDecodeError("Could not parse or repair JSON", content, 0)

            if result.get("condition_matched"):
                return EvalResult(
                    condition_matched=True,
                    matched_instruction_index=result.get("matched_instruction_index"),
                    reasoning=result.get("reasoning", ""),
                    action_json=result.get("action"),
                )
            else:
                return EvalResult(
                    condition_matched=False,
                    reasoning=result.get("reasoning", ""),
                )

        except httpx.HTTPStatusError as e:
            logger.error("LLM request failed with HTTP %d: %s", e.response.status_code, e)
            return EvalResult(condition_matched=False, reasoning=f"HTTP error: {e}")

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM JSON response: %s\nRaw: %s", e, content)
            return EvalResult(
                condition_matched=False,
                reasoning=f"JSON parse error: {e}",
            )

        except Exception as e:
            logger.error("Unexpected error during evaluation: %s", e, exc_info=True)
            return EvalResult(condition_matched=False, reasoning=f"Error: {e}")
