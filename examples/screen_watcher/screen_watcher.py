#!/usr/bin/env python3
"""
screen_watcher.py — GhostTunnel Client Example

This example script captures a screenshot of your screen at a regular interval (default: 30 seconds)
and sends it to ChatGPT Vision via a running GhostTunnel proxy server to get a description.

On macOS, it uses the macOS Accessibility API (via PyObjC AXUIElement) to detect structural UI
changes (app switching, window focus changes, modal dialogs) to avoid unnecessary API calls when
the UI is unchanged (ignoring minor text typing, cursor blinks, and terminal scrolling).

Requirements:
  - PIL (Pillow) — already in requirements.txt
  - httpx — already in requirements.txt
  - pyobjc-framework-ApplicationServices (macOS) — already in requirements.txt
  - A running GhostTunnel proxy server (default: http://127.0.0.1:8100)

Usage:
  python examples/screen_watcher/screen_watcher.py --interval 30 --skip --output watcher_log.jsonl
"""

import argparse
import asyncio
import base64
import datetime
from dataclasses import dataclass
import io
import json
import os
import sys
import time
from typing import Optional, Tuple, Dict, Any
import httpx
from PIL import ImageGrab, Image, ImageDraw, ImageFont

def draw_som_grid(image: Image.Image, grid_step: int = 50) -> Image.Image:
    """
    Superimpose a Set-of-Marks (SoM) 0-1000 coordinate grid on an image copy.
    Draws semi-transparent cyan grid lines and yellow coordinate numbers every 50 units.
    """
    grid_img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", grid_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = grid_img.size

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw vertical grid lines & yellow X labels (0-1000 scale)
    for x in range(0, w, grid_step):
        norm_x = int((x / w) * 1000)
        draw.line([(x, 0), (x, h)], fill=(0, 255, 255, 75), width=1)
        if norm_x % 50 == 0:
            draw.text((x + 2, 4), f"{norm_x}", fill=(255, 255, 0, 255), font=font, stroke_fill=(0, 0, 0, 255), stroke_width=1)

    # Draw horizontal grid lines & yellow Y labels (0-1000 scale)
    for y in range(0, h, grid_step):
        norm_y = int((y / h) * 1000)
        draw.line([(0, y), (w, y)], fill=(0, 255, 255, 75), width=1)
        if norm_y % 50 == 0:
            draw.text((4, y + 2), f"{norm_y}", fill=(255, 255, 0, 255), font=font, stroke_fill=(0, 0, 0, 255), stroke_width=1)

    return Image.alpha_composite(grid_img, overlay).convert("RGB")

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "click_hide_button",
            "description": "Click the hide/collapse button on the notification overlay panel header at normalized 0-1000 coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "Normalized X coordinate from 0 (left edge) to 1000 (right edge)"
                    },
                    "y": {
                        "type": "integer",
                        "description": "Normalized Y coordinate from 0 (top edge) to 1000 (bottom edge)"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of target element location"
                    }
                },
                "required": ["x", "y"]
            }
        }
    }
]

def crop_overlay_region(image: Image.Image) -> Tuple[Image.Image, int, int, int, int]:
    """
    Crop the lower-right region of the screen where floating overlay panels reside.
    Returns (cropped_image, crop_left, crop_top, crop_width, crop_height).
    """
    w, h = image.size
    crop_left = int(w * 0.60)  # Right 40% of screen
    crop_top = int(h * 0.15)   # Y from 15% to 85%
    crop_w = int(w * 0.40)
    crop_h = int(h * 0.70)

    cropped_img = image.crop((crop_left, crop_top, crop_left + crop_w, crop_top + crop_h))
    return cropped_img, crop_left, crop_top, crop_w, crop_h

def map_crop_normalized_to_screen(
    norm_x: int, norm_y: int,
    crop_left: int, crop_top: int, crop_w: int, crop_h: int
) -> Tuple[float, float]:
    """
    Map normalized (0-1000) coordinates from a cropped sub-image back to native macOS screen points.
    """
    if 0 < norm_x <= 100:
        norm_x *= 10
    if 0 < norm_y <= 100:
        norm_y *= 10

    rel_x = (float(norm_x) / 1000.0) * float(crop_w)
    rel_y = (float(norm_y) / 1000.0) * float(crop_h)

    screen_x = float(crop_left) + rel_x
    screen_y = float(crop_top) + rel_y
    return screen_x, screen_y

def map_normalized_to_screen(norm_x: int, norm_y: int) -> Tuple[float, float]:
    """
    Map normalized 0-1000 grid coordinates to macOS logical screen points.
    Auto-corrects if the model provided a 0-100 percentage scale (<= 100) instead of 0-1000 grid scale.
    """
    # Auto-correct 0-100 percentage values to 0-1000 grid scale
    if 0 < norm_x <= 100:
        norm_x *= 10
    if 0 < norm_y <= 100:
        norm_y *= 10

    if sys.platform == "darwin":
        try:
            from AppKit import NSScreen
            main_screen = NSScreen.mainScreen()
            if main_screen:
                frame = main_screen.frame()
                width = frame.size.width
                height = frame.size.height
                screen_x = (float(norm_x) / 1000.0) * width
                screen_y = (float(norm_y) / 1000.0) * height
                return screen_x, screen_y
        except Exception:
            pass
    return (float(norm_x) / 1000.0) * 1920.0, (float(norm_y) / 1000.0) * 1080.0

def click_screen_position(screen_x: float, screen_y: float):
    """
    Simulate a native macOS hardware left mouse click at logical screen coordinates.
    Warps the mouse pointer directly to (screen_x, screen_y) and posts mouse move, down, and up events to kCGSessionEventTap.
    """
    if sys.platform != "darwin":
        print(f"🖱️ Click simulation at ({screen_x:.1f}, {screen_y:.1f}) skipped (non-macOS system).")
        return
    try:
        from Quartz.CoreGraphics import (
            CGEventCreateMouseEvent,
            CGEventPost,
            CGWarpMouseCursorPosition,
            kCGEventLeftMouseDown,
            kCGEventLeftMouseUp,
            kCGEventMouseMoved,
            kCGSessionEventTap,
            CGPoint,
        )
        point = CGPoint(screen_x, screen_y)

        # 1. Warp mouse pointer directly to target screen coordinates
        CGWarpMouseCursorPosition(point)
        time.sleep(0.05)

        # 2. Post MouseMoved event to the user's GUI session
        move_evt = CGEventCreateMouseEvent(None, kCGEventMouseMoved, point, 0)
        CGEventPost(kCGSessionEventTap, move_evt)
        time.sleep(0.05)

        # 3. Post LeftMouseDown and LeftMouseUp events
        down_evt = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, point, 0)
        up_evt = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, point, 0)

        CGEventPost(kCGSessionEventTap, down_evt)
        time.sleep(0.05)
        CGEventPost(kCGSessionEventTap, up_evt)

        print(f"🖱️ [Auto-Click] Executed native mouse click at screen point ({screen_x:.1f}, {screen_y:.1f})")
    except Exception as e:
        print(f"❌ Failed to execute mouse click: {e}")

@dataclass(frozen=True)
class UISignature:
    """Represents a lightweight, hashable snapshot of the frontmost application's structural UI state."""
    app_bundle_id: str
    active_window_title: str
    window_count: int
    window_roles: Tuple[str, ...]
    modal_sheet_count: int
    element_tree_structure: Tuple[str, ...]

def check_macos_accessibility_permissions() -> bool:
    """
    Check if the process has macOS Accessibility permission (AXIsProcessTrusted).
    Returns True if granted or non-macOS, False if denied on macOS.
    """
    if sys.platform != "darwin":
        return True
    try:
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except ImportError:
        return True

def print_accessibility_permission_warning():
    """Print formatted instructions for granting Accessibility permission on macOS."""
    print("\n" + "=" * 65)
    print("❌ ERROR: macOS Accessibility Permission Denied")
    print("=" * 65)
    print("GhostTunnel screen_watcher requires macOS Accessibility permissions")
    print("to observe structural application state changes.")
    print("\nTo grant permissions:")
    print("  1. Open System Settings -> Privacy & Security -> Accessibility")
    print("  2. Toggle ON permission for your Terminal / IDE / Python app.")
    print("  3. Re-run this script.")
    print("=" * 65 + "\n")

def get_active_app_ui_signature(verbose: bool = False) -> Optional[UISignature]:
    """
    Query the active macOS application's Accessibility UI tree using PyObjC.
    Returns a UISignature object representing the current structural UI state.
    """
    if sys.platform != "darwin":
        return None
    try:
        from AppKit import NSWorkspace
        from ApplicationServices import (
            AXUIElementCreateApplication,
            AXUIElementCopyAttributeValue,
            AXUIElementSetAttributeValue,
            kAXFocusedWindowAttribute,
            kAXTitleAttribute,
            kAXSubroleAttribute,
            kAXRoleAttribute,
            kAXChildrenAttribute,
        )

        frontmost_app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if not frontmost_app:
            return UISignature("unknown", "", 0, (), 0, ())

        bundle_id = frontmost_app.bundleIdentifier() or frontmost_app.localizedName() or "unknown"
        pid = frontmost_app.processIdentifier()

        app_ax = AXUIElementCreateApplication(pid)
        try:
            AXUIElementSetAttributeValue(app_ax, "AXManualAccessibility", True)
        except Exception:
            pass

        # 1. AXWindows attribute inspection (Windows)
        win_roles = []
        err, windows = AXUIElementCopyAttributeValue(app_ax, "AXWindows", None)
        if err == 0 and windows:
            for w in windows:
                _, r = AXUIElementCopyAttributeValue(w, kAXRoleAttribute, None)
                _, s = AXUIElementCopyAttributeValue(w, kAXSubroleAttribute, None)
                r_str = f"{r}:{s}" if s else str(r) if r else "AXWindow"
                win_roles.append(r_str)

        # 2. Focused Window / Main Window Title
        err, focused_win = AXUIElementCopyAttributeValue(app_ax, kAXFocusedWindowAttribute, None)
        if err != 0 or not focused_win:
            err, focused_win = AXUIElementCopyAttributeValue(app_ax, "AXMainWindow", None)

        window_title = ""
        if focused_win:
            err_title, title_val = AXUIElementCopyAttributeValue(focused_win, kAXTitleAttribute, None)
            if err_title == 0 and title_val:
                window_title = str(title_val)

        # 3. Element Tree Traversal (Modals/Overlays & TreeNodes)
        modal_sheet_count = 0
        tree_roles = []
        modal_keywords = {
            "AXSheet", "AXDialog", "AXPopover", "AXDrawer", 
            "AXSystemDialog", "AXFloatingWindow", "AXAlert", "AXNotification",
            "AXApplicationAlert", "AXWebApplication", "AXLandmarkRegion"
        }

        def traverse(elem, depth=0, max_depth=10):
            nonlocal modal_sheet_count
            if not elem or depth > max_depth:
                return
            err, children = AXUIElementCopyAttributeValue(elem, kAXChildrenAttribute, None)
            if err != 0 or not children:
                return
            for child in children:
                err_r, r = AXUIElementCopyAttributeValue(child, kAXRoleAttribute, None)
                err_s, s = AXUIElementCopyAttributeValue(child, kAXSubroleAttribute, None)
                r_str = str(r) if r else ""
                s_str = str(s) if s else ""
                if r_str in modal_keywords or s_str in modal_keywords:
                    modal_sheet_count += 1
                role_key = f"{r_str}:{s_str}" if s_str else r_str
                if role_key:
                    tree_roles.append(role_key)
                traverse(child, depth + 1, max_depth)

        if focused_win:
            traverse(focused_win)

        # Also inspect top-level app children
        err, app_children = AXUIElementCopyAttributeValue(app_ax, kAXChildrenAttribute, None)
        if err == 0 and app_children:
            for child in app_children:
                err_r, r = AXUIElementCopyAttributeValue(child, kAXRoleAttribute, None)
                err_s, s = AXUIElementCopyAttributeValue(child, kAXSubroleAttribute, None)
                r_str = str(r) if r else ""
                s_str = str(s) if s else ""
                if r_str in modal_keywords or s_str in modal_keywords:
                    modal_sheet_count += 1
                role_key = f"{r_str}:{s_str}" if s_str else r_str
                if role_key:
                    tree_roles.append(role_key)

        sig = UISignature(
            app_bundle_id=bundle_id,
            active_window_title=window_title,
            window_count=len(windows) if windows else 0,
            window_roles=tuple(sorted(win_roles)),
            modal_sheet_count=modal_sheet_count,
            element_tree_structure=tuple(sorted(tree_roles)),
        )

        if verbose:
            print(f"🔍 [AX State] App: {sig.app_bundle_id} | Window: '{sig.active_window_title}' | Windows: {sig.window_count} | Modals/Overlays: {sig.modal_sheet_count} | TreeNodes: {len(sig.element_tree_structure)}")

        return sig
    except Exception as e:
        if verbose:
            print(f"⚠️ Failed to query AX UI signature: {e}")
        return None

def images_are_similar(img1: Image.Image, img2: Image.Image, threshold: float = 25.0) -> tuple[bool, float, float]:
    """
    Fallback visual pixel diff for non-macOS platforms.
    Compare two PIL Images. Returns (is_similar, global_mae, max_block_mae).
    """
    try:
        from PIL import ImageFilter
        w, h = img1.size
        left = 0
        right = w
        top = int(h * 0.05)
        bottom = int(h * 0.60)
        
        c1 = img1.crop((left, top, right, bottom))
        c2 = img2.crop((left, top, right, bottom))

        size = (512, 160)
        g1 = c1.convert("L").resize(size, Image.Resampling.BILINEAR)
        g2 = c2.convert("L").resize(size, Image.Resampling.BILINEAR)
        
        f1 = g1.filter(ImageFilter.MedianFilter(size=9))
        f2 = g2.filter(ImageFilter.MedianFilter(size=9))
        
        p1 = list(f1.tobytes())
        p2 = list(f2.tobytes())

        global_mae = sum(abs(a - b) for a, b in zip(p1, p2)) / len(p1)

        grid_cols, grid_rows = 8, 8
        block_w = size[0] // grid_cols
        block_h = size[1] // grid_rows

        block_maes = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                diffs = []
                for y in range(r * block_h, (r + 1) * block_h):
                    for x in range(c * block_w, (c + 1) * block_w):
                        idx = y * size[0] + x
                        diffs.append(abs(p1[idx] - p2[idx]))
                block_maes.append(sum(diffs) / len(diffs))

        max_block_mae = max(block_maes)
        global_threshold = threshold / 3.0
        is_similar = (global_mae < global_threshold) and (max_block_mae < threshold)
        return is_similar, global_mae, max_block_mae
    except Exception:
        return False, 0.0, 0.0

async def capture_screenshot() -> tuple[str, Image.Image]:
    """
    Capture a screenshot of the screen and return it as a tuple: (JPEG base64 Data URI, PIL Image).
    """
    try:
        screenshot = ImageGrab.grab()
        comparison_img = screenshot.copy()
        
        max_size = 1920
        width, height = screenshot.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)
        
        if screenshot.mode in ("RGBA", "P"):
            screenshot = screenshot.convert("RGB")
        buf = io.BytesIO()
        screenshot.save(buf, format="JPEG", quality=75)
        raw_bytes = buf.getvalue()
        
        b64_data = base64.b64encode(raw_bytes).decode("utf-8")
        size_kb = len(raw_bytes) / 1024
        print(f"📸 Captured screenshot ({screenshot.width}x{screenshot.height}, {size_kb:.1f} KB)")
        return f"data:image/jpeg;base64,{b64_data}", comparison_img
        
    except OSError as e:
        print(f"❌ OSError during screen capture: {e}")
        print("💡 TIP: On macOS, ensure your terminal/editor has Screen Recording permission under System Settings.")
        raise
    except Exception as e:
        print(f"❌ Failed to capture screenshot: {e}")
        raise

async def get_description_stream(server_url: str, model: str, image_uri: str, prompt: str, tools: Optional[list] = None) -> Tuple[Optional[str], list]:
    """
    Send the base64 image URI to GhostTunnel completions endpoint and stream the result.
    Returns (full_text_description, tool_calls_list).
    """
    endpoint = f"{server_url.rstrip('/')}/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_uri}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "stream": True
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    headers = {"Content-Type": "application/json"}
    print("🤖 Prompting ChatGPT Vision...", end="", flush=True)
    
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    try:
                        err_body = await resp.aread()
                        err_msg = err_body.decode()
                    except Exception:
                        err_msg = f"HTTP {resp.status_code}"
                    print(f"\n❌ Server error: {err_msg}")
                    return None, []
                
                print(" Receiving stream:\n")
                full_text = ""
                tool_calls_acc = {}
                
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "error" in chunk:
                                print(f"\n❌ Stream error: {chunk['error']}")
                                return None, []
                            
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content
                                print(content, end="", flush=True)
                                
                            raw_tc = delta.get("tool_calls", [])
                            for tc in raw_tc:
                                idx = tc.get("index", 0)
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {"name": "", "arguments": ""}
                                fn = tc.get("function", {})
                                if fn.get("name"):
                                    tool_calls_acc[idx]["name"] = fn["name"]
                                if fn.get("arguments"):
                                    tool_calls_acc[idx]["arguments"] += fn["arguments"]
                        except json.JSONDecodeError:
                            pass
                print("\n")
                
                parsed_tool_calls = []
                for idx in sorted(tool_calls_acc.keys()):
                    tc_info = tool_calls_acc[idx]
                    try:
                        args_obj = json.loads(tc_info["arguments"]) if tc_info["arguments"] else {}
                    except json.JSONDecodeError:
                        args_obj = {}
                    parsed_tool_calls.append({
                        "name": tc_info["name"],
                        "arguments": args_obj
                    })
                    
                return full_text, parsed_tool_calls
                
        except httpx.ConnectError:
            print(f"\n❌ Connection failed! Is GhostTunnel server running at {server_url}?")
            return None, []
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return None, []

def save_to_log(log_path: str, timestamp: str, description: str):
    """Save the description to a local JSONL log file."""
    log_entry = {
        "timestamp": timestamp,
        "description": description
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to write to log file: {e}")

def extract_target_coordinates(description_text: Optional[str], tool_calls: list) -> Tuple[Optional[int], Optional[int], str]:
    """
    Extract target (x, y) normalized coordinates from either native tool_calls
    or text responses containing coordinate predictions.
    Returns (norm_x, norm_y, source_info).
    """
    # 1. Native API tool calls
    for tc in tool_calls:
        if tc.get("name") == "click_hide_button":
            args_obj = tc.get("arguments", {})
            x = args_obj.get("x")
            y = args_obj.get("y")
            if x is not None and y is not None:
                return int(x), int(y), args_obj.get("reasoning", "Native Tool Call")

    if not description_text:
        return None, None, ""

    import re
    # 2. JSON pattern: {"x": 985, "y": 305}
    json_match = re.search(r'\{\s*"x"\s*:\s*(\d{1,4})\s*,\s*"y"\s*:\s*(\d{1,4})\s*\}', description_text, re.IGNORECASE)
    if json_match:
        return int(json_match.group(1)), int(json_match.group(2)), "Text JSON"

    # 3. Explicit X=..., Y=... pattern
    xy_match = re.search(r'X\s*=\s*(\d{1,4})\s*,\s*Y\s*=\s*(\d{1,4})', description_text, re.IGNORECASE)
    if xy_match:
        return int(xy_match.group(1)), int(xy_match.group(2)), "Text X=,Y="

    # 4. Any numeric pair (X, Y) or X, Y where 0 <= X, Y <= 1000
    pairs = re.findall(r'(\d{1,4})\s*,\s*(\d{1,4})', description_text)
    for tx, ty in pairs:
        ix, iy = int(tx), int(ty)
        if 0 <= ix <= 1000 and 0 <= iy <= 1000:
            return ix, iy, "Text Pair"

    return None, None, ""

def image_to_base64_uri(image: Image.Image, max_size: int = 1920) -> str:
    """Convert a PIL Image to a JPEG base64 Data URI."""
    img = image.copy()
    width, height = img.size
    if width > max_size or height > max_size:
        ratio = min(max_size / width, max_size / height)
        new_size = (int(width * ratio), int(height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

async def main():
    parser = argparse.ArgumentParser(
        description="GhostTunnel screen watcher example using ChatGPT Vision with macOS Accessibility UI change detection."
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=30.0,
        help="Capture interval in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--server", "-s", type=str, default="http://127.0.0.1:8100",
        help="GhostTunnel proxy server URL (default: http://127.0.0.1:8100)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="auto",
        help="Model slug to use (default: auto)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Optional path to a JSONL file to log the descriptions"
    )
    parser.add_argument(
        "--count", "-n", type=int, default=0,
        help="Number of captures to run before exiting (default: 0 for unlimited)"
    )
    parser.add_argument(
        "--skip", action="store_true",
        help="Enable skipping API calls when the UI state is unchanged"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=25.0,
        help="Pixel diff threshold fallback for non-macOS systems (default: 25.0)"
    )
    parser.add_argument(
        "--verbose-ax", action="store_true",
        help="Enable verbose macOS Accessibility UI signature logging"
    )
    parser.add_argument(
        "--auto-click", action="store_true",
        help="Automatically execute native macOS mouse click when Vision predicts target overlay button coordinates"
    )
    parser.add_argument(
        "--som-grid", action="store_true",
        help="Superimpose Set-of-Marks (SoM) visual 0-1000 coordinate grid on screenshot for ChatGPT Vision"
    )
    parser.add_argument(
        "--prompt", "-p", type=str,
        default="Describe what's on my screen right now in a single, concise paragraph.",
        help="Prompt to pass alongside the screenshot"
    )
    
    args = parser.parse_args()
    
    # 1. Enforce macOS Accessibility permissions check if running on macOS
    if sys.platform == "darwin":
        if not check_macos_accessibility_permissions():
            print_accessibility_permission_warning()
            sys.exit(1)
    
    print("==================================================")
    print("   GhostTunnel Vision Watcher (macOS AX Mode)     ")
    print("==================================================")
    print(f"Interval:     {args.interval}s")
    print(f"Server:       {args.server}")
    print(f"Model:        {args.model}")
    print(f"Skip Mode:    {args.skip}")
    print(f"Auto-Click:   {args.auto_click}")
    print(f"SoM Grid:     {args.som_grid}")
    print(f"Detection:    {'macOS Accessibility (AXUIElement)' if sys.platform == 'darwin' else 'Pixel Diffing'}")
    if args.verbose_ax:
        print(f"Verbose AX:   True")
    if args.output:
        print(f"Log File:     {args.output}")
    print("Press Ctrl+C to stop.")
    print("==================================================")
    
    loop_count = 0
    last_signature: Optional[UISignature] = None
    last_screenshot: Optional[Image.Image] = None
    
    try:
        while True:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] (Iteration {loop_count + 1})")
            
            try:
                is_similar = False
                
                if sys.platform == "darwin":
                    curr_signature = get_active_app_ui_signature(verbose=args.verbose_ax)
                    if args.skip and last_signature is not None and curr_signature is not None:
                        is_similar = (curr_signature == last_signature)
                        if is_similar:
                            print(f"📷 UI structural state unchanged (App: {curr_signature.app_bundle_id}, Window: '{curr_signature.active_window_title}'). Skipping API call.")
                else:
                    # Fallback for non-macOS systems
                    curr_signature = None
                
                if is_similar:
                    pass
                else:
                    if sys.platform == "darwin" and curr_signature is not None:
                        last_signature = curr_signature

                    raw_img_uri, screenshot_img = await capture_screenshot()
                    
                    # Non-macOS fallback pixel similarity check
                    if sys.platform != "darwin" and args.skip and last_screenshot is not None:
                        is_similar, g_mae, b_mae = images_are_similar(last_screenshot, screenshot_img, args.threshold)
                        if is_similar:
                            print("📷 Screen unchanged (pixel diff). Skipping API call.")
                    
                    if not is_similar:
                        if sys.platform != "darwin":
                            last_screenshot = screenshot_img

                        tools_to_use = None
                        prompt_to_use = args.prompt
                        is_overlay_active = (sys.platform == "darwin" and curr_signature and curr_signature.modal_sheet_count > 0)
                        
                        crop_meta = None
                        if is_overlay_active:
                            cropped_img, c_left, c_top, c_w, c_h = crop_overlay_region(screenshot_img)
                            crop_meta = (c_left, c_top, c_w, c_h)
                            print(f"✂️ Cropped overlay panel region ({c_w}x{c_h} at [{c_left}, {c_top}]) for high-res Vision targeting")
                            
                            if args.som_grid:
                                vision_img = draw_som_grid(cropped_img)
                                print("🎨 Superimposed Set-of-Marks (SoM) 0-1000 grid on cropped overlay image")
                            else:
                                vision_img = cropped_img
                            img_uri = image_to_base64_uri(vision_img)
                        elif args.som_grid:
                            vision_img = draw_som_grid(screenshot_img)
                            print("🎨 Superimposed Set-of-Marks (SoM) 0-1000 visual grid overlay on screenshot")
                            img_uri = image_to_base64_uri(vision_img)
                        else:
                            img_uri = raw_img_uri

                        if is_overlay_active:
                            tools_to_use = TOOLS_SCHEMA
                            prompt_to_use = (
                                f"{args.prompt}\n\n"
                                "This image is a high-resolution zoomed-in crop of the NOTIFICATIONS panel overlay on screen.\n"
                                "Look at the top-right header bar of the NOTIFICATIONS panel where three action icons are displayed:\n"
                                "  1. Filter list icon (left)\n"
                                "  2. Bell icon with slash (middle)\n"
                                "  3. Down chevron / 'v' collapse icon (farthest right)\n"
                                "Predict the exact normalized 0-1000 (x, y) coordinates targeting icon #3 (the 'v' down chevron icon on the far-right edge of the panel header).\n"
                                "Call the `click_hide_button` tool or state the coordinates as: `normalized coordinates **(X, Y)**`."
                            )

                        description, tool_calls = await get_description_stream(
                            server_url=args.server,
                            model=args.model,
                            image_uri=img_uri,
                            prompt=prompt_to_use,
                            tools=tools_to_use
                        )
                        
                        # Process target coordinates from native tool call OR text output
                        norm_x, norm_y, source = extract_target_coordinates(description, tool_calls)
                        if norm_x is not None and norm_y is not None:
                            if crop_meta:
                                c_left, c_top, c_w, c_h = crop_meta
                                screen_x, screen_y = map_crop_normalized_to_screen(norm_x, norm_y, c_left, c_top, c_w, c_h)
                            else:
                                screen_x, screen_y = map_normalized_to_screen(norm_x, norm_y)

                            print(f"\n🎯 [Vision Target] Predicted hide button at normalized ({norm_x}, {norm_y}) [{source}] -> Screen ({screen_x:.1f}, {screen_y:.1f})")
                            
                            if args.auto_click:
                                click_screen_position(screen_x, screen_y)
                            else:
                                print("💡 [Dry Run] Pass --auto-click to automatically click target coordinates.")
                        
                        if description and args.output:
                            save_to_log(args.output, timestamp, description)
                    
            except Exception as e:
                print(f"❌ Error during iteration: {e}")
                import traceback
                traceback.print_exc()
                
            loop_count += 1
            if args.count > 0 and loop_count >= args.count:
                print(f"Target count of {args.count} iterations reached. Exiting.")
                break
                
            print(f"Sleeping for {args.interval} seconds...")
            await asyncio.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nWatcher stopped by user. Goodbye!")
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
