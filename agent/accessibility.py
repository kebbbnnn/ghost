"""macOS Accessibility (AXUIElement) wrapper.

Queries the accessibility tree to find window bounds and UI state.
Requires Accessibility permission in System Settings.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard: Accessibility APIs are only available on macOS
# ---------------------------------------------------------------------------
_AX_AVAILABLE = False

if platform.system() == "Darwin":
    try:
        from ApplicationServices import (  # type: ignore[import-untyped]
            AXUIElementCreateApplication,
            AXUIElementCopyAttributeValue,
            AXValueGetValue,
            kAXWindowsAttribute,
            kAXPositionAttribute,
            kAXSizeAttribute,
            kAXTitleAttribute,
            kAXFocusedUIElementAttribute,
            kAXRoleAttribute,
            kAXValueAttribute,
            kAXChildrenAttribute,
        )
        from AppKit import NSWorkspace  # type: ignore[import-untyped]
        from Quartz import (  # type: ignore[import-untyped]
            kCGWindowListOptionOnScreenOnly,
            CGWindowListCopyWindowInfo,
            kCGNullWindowID,
        )

        _AX_AVAILABLE = True
    except ImportError:
        logger.warning(
            "PyObjC ApplicationServices/AppKit bindings not installed. "
            "Install pyobjc-framework-ApplicationServices and pyobjc-framework-Cocoa."
        )
else:
    logger.warning(
        "Accessibility module requires macOS. Current platform: %s",
        platform.system(),
    )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WindowInfo:
    """Geometry and identity of an application window."""

    app_name: str
    window_title: str
    x: int
    y: int
    width: int
    height: int
    pid: int


@dataclass(frozen=True, slots=True)
class AccessibilitySnapshot:
    """Lightweight snapshot of an application's accessibility state."""

    app_name: str
    window_title: str
    focused_element: dict | None  # keys: role, title, value
    visible_buttons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_ax_attribute(element: object, attribute: str) -> object | None:
    """Safely read a single AX attribute, returning *None* on any error."""
    err, value = AXUIElementCopyAttributeValue(element, attribute, None)  # type: ignore[arg-type]
    if err != 0:
        return None
    return value


def _extract_point(ax_value: object) -> tuple[int, int] | None:
    """Extract (x, y) from an AXValue wrapping a CGPoint."""
    from CoreFoundation import CFGetTypeID  # type: ignore[import-untyped]

    try:
        ok, point = AXValueGetValue(ax_value, 1, None)  # 1 = kAXValueCGPointType
        if ok:
            return (int(point.x), int(point.y))
    except Exception:
        pass
    return None


def _extract_size(ax_value: object) -> tuple[int, int] | None:
    """Extract (width, height) from an AXValue wrapping a CGSize."""
    try:
        ok, size = AXValueGetValue(ax_value, 2, None)  # 2 = kAXValueCGSizeType
        if ok:
            return (int(size.width), int(size.height))
    except Exception:
        pass
    return None


def _find_running_app(app_name: str) -> object | None:
    """Find a running application by its localized name (case-insensitive).

    Returns the ``NSRunningApplication`` instance or *None*.
    """
    workspace = NSWorkspace.sharedWorkspace()
    for app in workspace.runningApplications():
        localized = app.localizedName()
        if localized and localized.lower() == app_name.lower():
            return app
    return None


def _collect_buttons(element: object, depth: int = 0, max_depth: int = 5) -> list[str]:
    """Recursively walk AX children to find buttons and return their titles."""
    if depth > max_depth:
        return []

    buttons: list[str] = []

    role = _get_ax_attribute(element, kAXRoleAttribute)
    if role == "AXButton":
        title = _get_ax_attribute(element, kAXTitleAttribute)
        if title:
            buttons.append(str(title))

    children = _get_ax_attribute(element, kAXChildrenAttribute)
    if children:
        for child in children:
            buttons.extend(_collect_buttons(child, depth + 1, max_depth))

    return buttons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def find_window(app_name: str) -> WindowInfo | None:
    """Locate the frontmost window of *app_name* and return its geometry.

    Args:
        app_name: The application's user-visible name (e.g. ``"Safari"``).
            Matching is case-insensitive.

    Returns:
        A :class:`WindowInfo` with the window's screen-absolute bounds,
        or *None* if the application or its window could not be found.
    """
    if not _AX_AVAILABLE:
        logger.error("Accessibility APIs not available — cannot find window")
        return None

    app = _find_running_app(app_name)
    if app is None:
        logger.warning("Application '%s' not found among running apps", app_name)
        return None

    pid = app.processIdentifier()
    logger.debug("Found '%s' with PID %d", app_name, pid)

    # Get the AX application element and its first window
    ax_app = AXUIElementCreateApplication(pid)
    windows = _get_ax_attribute(ax_app, kAXWindowsAttribute)

    if not windows or len(windows) == 0:
        logger.warning("Application '%s' (PID %d) has no AX windows", app_name, pid)
        return None

    window = windows[0]

    # Title
    title = _get_ax_attribute(window, kAXTitleAttribute) or ""

    # Position (CGPoint wrapped in AXValue)
    pos_value = _get_ax_attribute(window, kAXPositionAttribute)
    if pos_value is None:
        logger.warning("Could not read window position for '%s'", app_name)
        return None
    point = _extract_point(pos_value)
    if point is None:
        logger.warning("Could not decode CGPoint for '%s' window position", app_name)
        return None

    # Size (CGSize wrapped in AXValue)
    size_value = _get_ax_attribute(window, kAXSizeAttribute)
    if size_value is None:
        logger.warning("Could not read window size for '%s'", app_name)
        return None
    size = _extract_size(size_value)
    if size is None:
        logger.warning("Could not decode CGSize for '%s' window size", app_name)
        return None

    info = WindowInfo(
        app_name=app_name,
        window_title=str(title),
        x=point[0],
        y=point[1],
        width=size[0],
        height=size[1],
        pid=pid,
    )
    logger.info(
        "Window: '%s' — '%s' at (%d, %d) %dx%d",
        info.app_name,
        info.window_title,
        info.x,
        info.y,
        info.width,
        info.height,
    )
    return info


def get_ui_snapshot(app_name: str) -> AccessibilitySnapshot | None:
    """Capture a lightweight accessibility snapshot of *app_name*.

    Reads the focused UI element (role, title, value) and enumerates
    visible buttons in the first window.

    Args:
        app_name: The application's user-visible name.

    Returns:
        An :class:`AccessibilitySnapshot`, or *None* if the app could
        not be found or accessibility permissions are missing.
    """
    if not _AX_AVAILABLE:
        logger.error("Accessibility APIs not available — cannot snapshot")
        return None

    app = _find_running_app(app_name)
    if app is None:
        logger.warning("Application '%s' not found among running apps", app_name)
        return None

    pid = app.processIdentifier()
    ax_app = AXUIElementCreateApplication(pid)

    # ---- Window title ----
    windows = _get_ax_attribute(ax_app, kAXWindowsAttribute)
    window_title = ""
    if windows and len(windows) > 0:
        window_title = str(_get_ax_attribute(windows[0], kAXTitleAttribute) or "")

    # ---- Focused element ----
    focused: dict | None = None
    focused_el = _get_ax_attribute(ax_app, kAXFocusedUIElementAttribute)
    if focused_el is not None:
        role = _get_ax_attribute(focused_el, kAXRoleAttribute)
        title = _get_ax_attribute(focused_el, kAXTitleAttribute)
        value = _get_ax_attribute(focused_el, kAXValueAttribute)
        focused = {
            "role": str(role) if role else None,
            "title": str(title) if title else None,
            "value": str(value) if value else None,
        }
        logger.debug("Focused element: %s", focused)
    else:
        logger.debug("No focused element for '%s'", app_name)

    # ---- Visible buttons ----
    visible_buttons: list[str] = []
    if windows and len(windows) > 0:
        try:
            visible_buttons = _collect_buttons(windows[0])
        except Exception as exc:
            logger.warning("Failed to enumerate buttons for '%s': %s", app_name, exc)

    snapshot = AccessibilitySnapshot(
        app_name=app_name,
        window_title=window_title,
        focused_element=focused,
        visible_buttons=visible_buttons,
    )
    logger.info(
        "Accessibility snapshot for '%s': title='%s', focused=%s, %d button(s)",
        snapshot.app_name,
        snapshot.window_title,
        "yes" if snapshot.focused_element else "no",
        len(snapshot.visible_buttons),
    )
    return snapshot
