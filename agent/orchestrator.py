"""Main observation loop orchestrator.

Coordinates the full agent pipeline:
  1. Capture target window
  2. Change gate (SSIM)
  3. Enrich state (OCR + Accessibility)
  4. Evaluate conditions (LLM)
  5. Compile + validate + execute action
  6. Post-action verification
  7. Loop
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from PIL import Image

from agent.config import (
    CAPTURE_INTERVAL,
    COOLDOWN_SECONDS,
    POST_ACTION_WAIT,
    SSIM_NO_CHANGE_THRESHOLD,
    PERSISTENCE_FRAMES,
    PROXY_URL,
    PROXY_MODEL,
)
from agent.frame_provider import WindowFrameProvider, FrameProvider
from agent.detector import ChangeDetector
from agent.ocr import extract_text, format_ocr_for_prompt
from agent.accessibility import get_ui_snapshot, find_window
from agent.evaluator import ConditionEvaluator
from agent.actions import parse_action, validate_action, compile_action

logger = logging.getLogger(__name__)


@dataclass
class AgentStats:
    """Runtime statistics for the observation agent."""

    frames_captured: int = 0
    changes_detected: int = 0
    llm_calls: int = 0
    actions_executed: int = 0
    actions_rejected: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def uptime(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> str:
        return (
            f"Uptime: {self.uptime:.0f}s | "
            f"Frames: {self.frames_captured} | "
            f"Changes: {self.changes_detected} | "
            f"LLM calls: {self.llm_calls} | "
            f"Actions: {self.actions_executed} | "
            f"Rejected: {self.actions_rejected} | "
            f"Errors: {self.errors}"
        )


class ObservationAgent:
    """Persistent looping agent that observes a target app and acts on instructions.

    Args:
        app_name: Name of the target application (e.g., "Cursor", "Chrome").
        instructions: List of condition -> action rules in natural language.
        proxy_url: URL of the Ghost proxy.
        model: Model name to use for LLM calls.
        frame_provider: Optional custom FrameProvider (defaults to WindowFrameProvider).
    """

    def __init__(
        self,
        app_name: str,
        instructions: list[str],
        proxy_url: str = PROXY_URL,
        model: str = PROXY_MODEL,
        frame_provider: FrameProvider | None = None,
    ):
        self.app_name = app_name
        self.instructions = instructions
        self.proxy_url = proxy_url
        self.model = model

        # Pipeline components
        self.frame_provider = frame_provider or WindowFrameProvider(app_name)
        self.detector = ChangeDetector(
            ssim_threshold=SSIM_NO_CHANGE_THRESHOLD,
            persistence_frames=PERSISTENCE_FRAMES,
        )
        self.evaluator = ConditionEvaluator(
            instructions=instructions,
            proxy_url=proxy_url,
            model=model,
        )

        # State
        self.stats = AgentStats()
        self._cooldown_until: float = 0.0
        self._running: bool = False

    def _is_on_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    def _start_cooldown(self) -> None:
        self._cooldown_until = time.time() + COOLDOWN_SECONDS
        logger.debug("Cooldown started until %.1f", self._cooldown_until)

    async def run(self) -> None:
        """Run the observation loop indefinitely until stopped."""
        self._running = True
        logger.info(
            "🔍 Agent started — observing '%s' with %d instruction(s)",
            self.app_name,
            len(self.instructions),
        )
        for i, inst in enumerate(self.instructions):
            logger.info("  [%d] %s", i, inst)

        try:
            while self._running:
                await self._tick()
                await asyncio.sleep(CAPTURE_INTERVAL)
        except asyncio.CancelledError:
            logger.info("Agent loop cancelled.")
        finally:
            self._running = False
            logger.info("Agent stopped. %s", self.stats.summary())

    def stop(self) -> None:
        """Signal the agent loop to stop."""
        self._running = False

    async def _tick(self) -> None:
        """Execute one observation cycle."""
        # 1. Capture frame
        frame = self.frame_provider.capture()
        if frame is None:
            logger.debug("No frame captured (window not found or minimized). Skipping.")
            return
        self.stats.frames_captured += 1

        # 2. Change gate
        result = self.detector.detect(frame)
        if not result.changed:
            return

        if not self.detector.should_proceed():
            return  # Change not yet persistent

        self.stats.changes_detected += 1

        # 3. Check cooldown
        if self._is_on_cooldown():
            logger.debug("On cooldown — skipping LLM evaluation.")
            return

        # 4. Enrich state
        ocr_text = ""
        ax_snapshot = {}

        try:
            ocr_results = extract_text(frame)
            ocr_text = format_ocr_for_prompt(ocr_results, image_size=frame.size)
        except Exception as e:
            logger.warning("OCR failed: %s", e)
            self.stats.errors += 1

        try:
            ax = get_ui_snapshot(self.app_name)
            if ax is not None:
                ax_snapshot = {
                    "app_name": ax.app_name,
                    "window_title": ax.window_title,
                    "focused_element": ax.focused_element,
                    "visible_buttons": ax.visible_buttons,
                }
        except Exception as e:
            logger.warning("Accessibility snapshot failed: %s", e)
            self.stats.errors += 1

        # 5. Evaluate conditions via LLM
        logger.info("📡 Evaluating conditions via LLM...")
        self.stats.llm_calls += 1

        eval_result = await self.evaluator.evaluate(
            screenshot=frame,
            ocr_text=ocr_text,
            ax_snapshot=ax_snapshot,
        )

        if not eval_result.condition_matched:
            logger.info(
                "No condition matched. Reason: %s",
                eval_result.reasoning[:100],
            )
            self._start_cooldown()
            return

        logger.info(
            "✅ Condition matched! Instruction[%s]: %s",
            eval_result.matched_instruction_index,
            eval_result.reasoning[:100],
        )

        # 6. Parse + validate + compile + execute action
        if eval_result.action_json is None:
            logger.warning("Condition matched but no action returned.")
            self._start_cooldown()
            return

        try:
            action = parse_action(eval_result.action_json)
        except Exception as e:
            logger.error("Failed to parse action JSON: %s — %s", eval_result.action_json, e)
            self.stats.errors += 1
            self._start_cooldown()
            return

        # Get window bounds for validation and coordinate translation
        window_info = find_window(self.app_name)
        if window_info is None:
            logger.error("Cannot find window for action execution.")
            self.stats.errors += 1
            self._start_cooldown()
            return

        valid, reason = validate_action(
            action,
            window_width=window_info.width,
            window_height=window_info.height,
        )
        if not valid:
            logger.warning("Action rejected by validator: %s — %s", action, reason)
            self.stats.actions_rejected += 1
            self._start_cooldown()
            return

        # Compile and execute
        executable = compile_action(
            action,
            window_x=window_info.x,
            window_y=window_info.y,
        )

        logger.info("🎯 Executing action: %s", eval_result.action_json)
        try:
            executable()
            self.stats.actions_executed += 1
        except Exception as e:
            logger.error("Action execution failed: %s", e, exc_info=True)
            self.stats.errors += 1
            self._start_cooldown()
            return

        # 7. Post-action verification
        logger.info("⏳ Waiting %.1fs for post-action verification...", POST_ACTION_WAIT)
        await asyncio.sleep(POST_ACTION_WAIT)

        post_frame = self.frame_provider.capture()
        if post_frame is not None:
            post_result = self.detector.detect(post_frame)
            if not post_result.changed:
                logger.warning(
                    "⚠️ Screen did not change after action (SSIM=%.4f). "
                    "Action may have failed.",
                    post_result.ssim,
                )
            else:
                logger.info(
                    "✔️ Screen changed after action (SSIM=%.4f). Action likely succeeded.",
                    post_result.ssim,
                )

        # Reset detector buffer after action to avoid re-triggering on stale state
        self.detector.reset()
        self._start_cooldown()
