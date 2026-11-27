"""
uploader.py — Image upload lifecycle for ChatGPT multimodal messages.

Handles the 3-step ChatGPT file upload protocol:
  1. Register file metadata via POST /backend-anon/files
  2. Upload raw bytes to Azure Blob presigned URL
  3. Return image_asset_pointer dicts for conversation payloads

This module is stateless — it receives the HTTP session and headers
as arguments rather than reaching into the provider's internals.
"""
import base64
import io
import logging
import re
import uuid
from typing import Any

import httpx
from curl_cffi.requests import AsyncSession
from ghost.core.exceptions import RateLimitError, TokenExpiredError

log = logging.getLogger(__name__)

BASE_URL = "https://chatgpt.com"
FILES_URL = f"{BASE_URL}/backend-anon/files"

# Regex for data URI: data:image/png;base64,iVBOR...
DATA_URI_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)

# Map MIME types to common file extensions
MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/svg+xml": ".svg",
}


async def preprocess_images(
    messages: list[dict],
    session: AsyncSession,
    base_headers: dict,
    oai_did: str,
    sentinel_headers: dict | None = None,
) -> tuple[list[dict], bool]:
    """
    Scan messages for image_url content parts, upload them to ChatGPT's
    file service, and return transformed messages.

    Args:
        messages:     OpenAI-format message list
        session:      The provider's curl_cffi AsyncSession (for /backend-anon calls)
        base_headers: Base browser-like headers (UA, sec-ch-ua, etc.)
        oai_did:      The current oai-device-id

    Returns:
        (transformed_messages, has_images)
        - transformed_messages: messages with image_url parts replaced by
          image_asset_pointer dicts, and attachment metadata injected
        - has_images: True if any images were found and uploaded
    """
    has_images = False
    transformed = []

    for msg in messages:
        content = msg.get("content")
        if msg.get("role") != "user" or not isinstance(content, list):
            transformed.append(msg)
            continue

        new_parts = []
        attachments = []

        for part in content:
            if isinstance(part, str):
                new_parts.append(part)
                continue

            if not isinstance(part, dict):
                new_parts.append(part)
                continue

            if part.get("type") != "image_url":
                new_parts.append(part)
                continue

            # --- This is an image_url part ---
            url_data = part.get("image_url", {})
            url = url_data.get("url", "") if isinstance(url_data, dict) else str(url_data)
            if not url:
                new_parts.append(part)
                continue

            has_images = True
            log.info("Processing image_url part (len=%d chars)", len(url[:50]))

            # Step 1: Resolve image bytes
            image_bytes, mime_type, filename = await _resolve_image(url, session)
            file_size = len(image_bytes)

            # Step 2: Get image dimensions
            width, height = _get_image_dimensions(image_bytes)

            # Step 3: Register file with ChatGPT
            headers = {
                **base_headers,
                "oai-device-id": oai_did,
                "oai-language": "en-US",
                "content-type": "application/json",
                **(sentinel_headers or {}),
            }
            file_id, upload_url = await _register_file(
                session, headers, filename, file_size
            )
            log.info("Registered file: %s (id=%s)", filename, file_id)

            # Step 4: Upload bytes to Azure Blob
            await _upload_blob(upload_url, image_bytes, mime_type)
            log.info("Uploaded %d bytes to blob storage", file_size)

            # Step 5: Finalize file by calling process_upload_stream
            # This tells the backend the file is ready for use
            process_url = f"{BASE_URL}/backend-anon/files/process_upload_stream"
            process_body = {
                "file_id": file_id,
                "use_case": "multimodal",
                "index_for_retrieval": False,
                "file_name": filename,
                "entry_surface": "chat_composer",
            }
            process_resp = await session.post(process_url, headers=headers, json=process_body)
            if process_resp.status_code == 200:
                log.info("File processing confirmed for %s", file_id)
            else:
                log.warning("process_upload_stream returned %d: %s", 
                           process_resp.status_code, process_resp.text)

            # Step 6: Build the image_asset_pointer part
            new_parts.append({
                "content_type": "image_asset_pointer",
                "asset_pointer": f"file-service://{file_id}",
                "size_bytes": file_size,
                "width": width,
                "height": height,
            })

            attachments.append({
                "id": file_id,
                "size": file_size,
                "name": filename,
                "mime_type": mime_type,
                "width": width,
                "height": height,
                "source": "local",
                "is_big_paste": False,
            })

        # Build the transformed message
        new_msg = {**msg, "content": new_parts}
        if attachments:
            # Inject attachment metadata (ChatGPT expects this on user messages)
            metadata = msg.get("metadata", {})
            metadata["attachments"] = attachments
            metadata["selected_github_repos"] = []
            metadata["selected_all_github_repos"] = False
            metadata["serialization_metadata"] = {"custom_symbol_offsets": []}
            new_msg["metadata"] = metadata

        transformed.append(new_msg)

    return transformed, has_images


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

async def _resolve_image(
    url: str, session: AsyncSession
) -> tuple[bytes, str, str]:
    """
    Resolve an image URL to raw bytes.

    Handles:
      - data:image/...;base64,... URIs
      - External HTTP(S) URLs

    Returns:
        (raw_bytes, mime_type, filename)

    Raises:
        ValueError: If the image cannot be fetched or decoded.
    """
    # --- Base64 data URI ---
    match = DATA_URI_RE.match(url)
    if match:
        mime_type = match.group(1)
        b64_data = match.group(2)
        try:
            raw_bytes = base64.b64decode(b64_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image data: {e}") from e

        ext = MIME_TO_EXT.get(mime_type, ".bin")
        filename = f"image_{uuid.uuid4().hex[:8]}{ext}"
        return raw_bytes, mime_type, filename

    # --- External URL ---
    if url.startswith(("http://", "https://")):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except Exception as e:
            raise ValueError(f"Failed to fetch image from URL '{url}': {e}") from e

        raw_bytes = resp.content
        # Detect MIME type from Content-Type header, fallback to image/jpeg
        ct = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        mime_type = ct if ct.startswith("image/") else "image/jpeg"

        # Try to extract filename from URL path
        from urllib.parse import urlparse
        path = urlparse(url).path
        filename = path.rsplit("/", 1)[-1] if "/" in path else f"image_{uuid.uuid4().hex[:8]}"
        if not any(filename.endswith(ext) for ext in MIME_TO_EXT.values()):
            ext = MIME_TO_EXT.get(mime_type, ".jpg")
            filename = f"{filename}{ext}"

        return raw_bytes, mime_type, filename

    raise ValueError(f"Unsupported image URL scheme: {url[:80]}")


def _get_image_dimensions(data: bytes) -> tuple[int, int]:
    """
    Extract (width, height) from raw image bytes using Pillow.
    Returns (0, 0) if Pillow is unavailable or dimensions cannot be determined.
    """
    try:
        from PIL import Image
    except ImportError:
        log.warning("Pillow not installed; install 'ghost[vision]' to extract image dimensions")
        return 0, 0
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.size  # (width, height)
    except Exception as e:
        log.warning("Could not read image dimensions: %s", e)
        return 0, 0


async def _register_file(
    session: AsyncSession,
    headers: dict,
    filename: str,
    file_size: int,
) -> tuple[str, str]:
    """
    POST /backend-anon/files to register a file upload.

    Returns:
        (file_id, upload_url)

    Raises:
        ValueError: If registration fails.
    """
    body = {
        "file_name": filename,
        "file_size": file_size,
        "use_case": "multimodal",
        "timezone_offset_min": -480,
        "reset_rate_limits": False,
    }

    try:
        resp = await session.post(FILES_URL, headers=headers, json=body)
        if resp.status_code == 429:
            raise RateLimitError("ChatGPT file upload rate limit (429)")
        if resp.status_code in (401, 403):
            raise TokenExpiredError(f"ChatGPT file upload unauthorized ({resp.status_code})")
        resp.raise_for_status()
    except (RateLimitError, TokenExpiredError):
        raise
    except Exception as e:
        raise ValueError(f"Failed to register file '{filename}': {e}") from e

    data = resp.json()
    file_id = data.get("file_id")
    upload_url = data.get("upload_url")

    if not file_id or not upload_url:
        raise ValueError(f"Unexpected file registration response: {data}")

    return file_id, upload_url


async def _upload_blob(
    upload_url: str,
    data: bytes,
    mime_type: str,
) -> None:
    """
    PUT raw image bytes to the Azure Blob Storage presigned URL.

    Uses httpx instead of curl_cffi because the Azure Blob endpoint
    doesn't require browser TLS fingerprinting.

    Raises:
        ValueError: If the upload fails.
    """
    headers = {
        "x-ms-blob-type": "BlockBlob",
        "Content-Type": mime_type,
        "Content-Length": str(len(data)),
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.put(upload_url, content=data, headers=headers)
            resp.raise_for_status()
    except Exception as e:
        raise ValueError(f"Failed to upload blob: {e}") from e

    log.debug("Blob upload complete (status=%d)", resp.status_code)
