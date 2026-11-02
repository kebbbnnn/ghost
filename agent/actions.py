"""Action schema, compiler, and policy validator.

Defines the structured action types the LLM can return,
validates them against safety policies, and compiles them
to pyautogui calls.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Callable, Literal, Union

import pyautogui  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, TypeAdapter

from agent.config import BLACKLISTED_HOTKEYS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action models (discriminated union on the ``action`` field)
# ---------------------------------------------------------------------------


class ClickAction(BaseModel):
    """Single left-click at window-relative coordinates."""

    action: Literal["click"] = "click"
    x: int
    y: int


class DoubleClickAction(BaseModel):
    """Double left-click at window-relative coordinates."""

    action: Literal["double_click"] = "double_click"
    x: int
    y: int


class RightClickAction(BaseModel):
    """Right-click at window-relative coordinates."""

    action: Literal["right_click"] = "right_click"
    x: int
    y: int


class TypeAction(BaseModel):
    """Type a string of text at the current cursor position."""

    action: Literal["type"] = "type"
    text: str


class HotkeyAction(BaseModel):
    """Press a keyboard shortcut (e.g. ``["command", "c"]``)."""

    action: Literal["hotkey"] = "hotkey"
    keys: list[str]


class ScrollAction(BaseModel):
    """Scroll the mouse wheel at the given coordinates."""

    action: Literal["scroll"] = "scroll"
    x: int
    y: int
    direction: Literal["up", "down"]
    amount: int = 3


class WaitAction(BaseModel):
    """Pause for a given number of seconds before the next action."""

    action: Literal["wait"] = "wait"
    seconds: float = 1.0


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

ActionUnion = Annotated[
    Union[
        ClickAction,
        DoubleClickAction,
        RightClickAction,
        TypeAction,
        HotkeyAction,
        ScrollAction,
        WaitAction,
    ],
    Field(discriminator="action"),
]

_action_adapter: TypeAdapter[ActionUnion] = TypeAdapter(ActionUnion)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_action(data: dict) -> ActionUnion:
    """Validate a raw dict into a typed ``ActionUnion`` instance.

    Raises
    ------
    pydantic.ValidationError
        If *data* does not match any known action schema.
    """
    return _action_adapter.validate_python(data)


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def validate_action(
    action: ActionUnion,
    window_width: int,
    window_height: int,
) -> tuple[bool, str]:
    """Check *action* against safety & bounds policies.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` when the action is acceptable, or
        ``(False, reason)`` when it must be rejected.
    """
    # --- Coordinate bounds -------------------------------------------------
    if isinstance(action, (ClickAction, DoubleClickAction, RightClickAction, ScrollAction)):
        if not (0 <= action.x <= window_width):
            return False, (
                f"x={action.x} out of bounds [0, {window_width}]"
            )
        if not (0 <= action.y <= window_height):
            return False, (
                f"y={action.y} out of bounds [0, {window_height}]"
            )

    # --- Hotkey blacklist --------------------------------------------------
    if isinstance(action, HotkeyAction):
        normalised = [k.lower().strip() for k in action.keys]
        for blacklisted in BLACKLISTED_HOTKEYS:
            if normalised == [b.lower() for b in blacklisted]:
                return False, (
                    f"Hotkey {normalised} is blacklisted for safety"
                )

    # --- Type length -------------------------------------------------------
    if isinstance(action, TypeAction):
        if len(action.text) > 500:
            return False, (
                f"TypeAction text length {len(action.text)} exceeds 500 character limit"
            )

    # --- Wait cap ----------------------------------------------------------
    if isinstance(action, WaitAction):
        if action.seconds > 10:
            return False, (
                f"WaitAction seconds={action.seconds} exceeds 10 s limit"
            )

    return True, ""


# ---------------------------------------------------------------------------
# Compilation to pyautogui calls
# ---------------------------------------------------------------------------


def compile_action(
    action: ActionUnion,
    window_x: int,
    window_y: int,
) -> Callable[[], None]:
    """Compile *action* into a zero-argument callable that performs the
    corresponding ``pyautogui`` operation.

    Window-relative coordinates are translated to absolute screen
    coordinates by adding the *window_x* / *window_y* offset of the
    target window's top-left corner.
    """

    def _safe(fn: Callable[[], None]) -> Callable[[], None]:
        """Wrap *fn* to log execution failures while propagating to the caller."""

        def _wrapper() -> None:
            try:
                fn()
            except Exception:
                logger.exception("Failed to execute action %s", action)
                raise

        return _wrapper

    # --- Click -------------------------------------------------------------
    if isinstance(action, ClickAction):
        abs_x, abs_y = window_x + action.x, window_y + action.y

        @_safe
        def _click() -> None:
            logger.info("click(%d, %d)", abs_x, abs_y)
            pyautogui.click(abs_x, abs_y)

        return _click

    # --- Double click ------------------------------------------------------
    if isinstance(action, DoubleClickAction):
        abs_x, abs_y = window_x + action.x, window_y + action.y

        @_safe
        def _double_click() -> None:
            logger.info("doubleClick(%d, %d)", abs_x, abs_y)
            pyautogui.doubleClick(abs_x, abs_y)

        return _double_click

    # --- Right click -------------------------------------------------------
    if isinstance(action, RightClickAction):
        abs_x, abs_y = window_x + action.x, window_y + action.y

        @_safe
        def _right_click() -> None:
            logger.info("rightClick(%d, %d)", abs_x, abs_y)
            pyautogui.rightClick(abs_x, abs_y)

        return _right_click

    # --- Type --------------------------------------------------------------
    if isinstance(action, TypeAction):
        text = action.text

        @_safe
        def _type() -> None:
            logger.info("write(%r)", text[:80])
            pyautogui.write(text, interval=0.02)

        return _type

    # --- Hotkey ------------------------------------------------------------
    if isinstance(action, HotkeyAction):
        keys = list(action.keys)

        @_safe
        def _hotkey() -> None:
            logger.info("hotkey(%s)", ", ".join(keys))
            pyautogui.hotkey(*keys)

        return _hotkey

    # --- Scroll ------------------------------------------------------------
    if isinstance(action, ScrollAction):
        abs_x, abs_y = window_x + action.x, window_y + action.y
        clicks = action.amount if action.direction == "up" else -action.amount

        @_safe
        def _scroll() -> None:
            logger.info("scroll(%d, x=%d, y=%d)", clicks, abs_x, abs_y)
            pyautogui.scroll(clicks, abs_x, abs_y)

        return _scroll

    # --- Wait --------------------------------------------------------------
    if isinstance(action, WaitAction):
        secs = action.seconds

        @_safe
        def _wait() -> None:
            logger.info("wait(%.1fs)", secs)
            time.sleep(secs)

        return _wait

    # Unreachable for valid ActionUnion, but guard defensively.
    raise ValueError(f"Unknown action type: {type(action).__name__}")
