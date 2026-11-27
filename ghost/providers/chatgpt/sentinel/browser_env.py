"""
browser_env.py — Spoofed browser environment signals for the Sentinel SDK VM.

These values are intentionally realistic for a macOS/Chrome combination.
The VM reads these values to build a fingerprint token.
"""
import random
import time


class BrowserEnv:
    """Hardcoded browser signals that the Sentinel VM collects."""

    # navigator.*
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    )
    language: str = "en-US"
    languages: str = "en-US,en"
    hardware_concurrency: int = 16
    device_memory: int = 8
    vendor: str = "Google Inc."
    platform: str = "Win32"

    # screen.*
    screen_width: int = 1920
    screen_height: int = 1080

    # WebGL GPU fingerprint
    gpu_vendor: str = "Google Inc. (NVIDIA)"
    gpu_renderer: str = (
        "ANGLE (NVIDIA, NVIDIA GeForce RTX 4070 Laptop GPU/PCIe/SSE2, OpenGL 4.6)"
    )

    # Timezone
    timezone_offset: int = -480   # UTC+8 (Asia/Manila)
    timezone: str = "Asia/Manila"

    # Misc
    pixel_ratio: float = 2.0
    max_touch_points: int = 0

    # Performance
    @staticmethod
    def performance_now() -> float:
        """Monotonic timer in ms — simulated."""
        return (time.monotonic() * 1000) % 600000

    @staticmethod
    def time_origin() -> float:
        """performance.timeOrigin — roughly (now - perf.now)."""
        return time.time() * 1000 - BrowserEnv.performance_now()

    # SDK script URL (determines the XOR key lookup)
    sdk_script_url: str = "https://sentinel.openai.com/sentinel/20260219f9f6/sdk.js"

    # History length (window.history.length) — plausible value
    history_length: int = 5

    # document.location
    location_href: str = "https://chatgpt.com/"

    # data-build attribute on <html>
    data_build: str = ""

    # Keys checked by the VM
    # Number("ai" in window) etc. — all 0 for a plain Python env
    ai_in_window: int = 0
    create_prng_in_window: int = 0
    cache_in_window: int = 0
    data_in_window: int = 0
    solana_in_window: int = 0
    dump_in_window: int = 0
    install_trigger_in_window: int = 0

    # Navigator prototype random key (used by getConfig N() call)
    nav_proto_keys = [
        "userAgent", "language", "languages", "onLine", "cookieEnabled",
        "hardwareConcurrency", "deviceMemory", "maxTouchPoints",
        "vendor", "platform", "product", "productSub", "doNotTrack",
        "pdfViewerEnabled", "connection"
    ]

    @staticmethod
    def random_nav_proto_entry() -> str:
        key = random.choice(BrowserEnv.nav_proto_keys)
        nav_values = {
            "userAgent": BrowserEnv.user_agent,
            "language": BrowserEnv.language,
            "languages": BrowserEnv.languages,
            "onLine": "true",
            "cookieEnabled": "true",
            "hardwareConcurrency": str(BrowserEnv.hardware_concurrency),
            "deviceMemory": str(BrowserEnv.device_memory),
            "maxTouchPoints": str(BrowserEnv.max_touch_points),
            "vendor": BrowserEnv.vendor,
            "platform": BrowserEnv.platform,
            "product": "Gecko",
            "productSub": "20030107",
            "doNotTrack": "null",
            "pdfViewerEnabled": "true",
            "connection": "[object NetworkInformation]",
        }
        return f"{key}−{nav_values.get(key, key)}"
