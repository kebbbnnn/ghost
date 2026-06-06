"""
video_processor.py — Server-side video frame extraction with perceptual hash
deduplication for video understanding.

Pipeline:
  1. Save video to temp file (OpenCV requires filesystem access)
  2. Sample frames at configurable FPS rate
  3. Compute perceptual hash (pHash) for each frame
  4. Deduplicate: keep frames whose Hamming distance from last kept frame >= threshold
  5. Resize surviving keyframes to max dimension
  6. Encode as JPEG base64 data URIs with timestamp annotations
"""

import base64
import logging
import os
import tempfile
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


def compute_phash(frame_bgr: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """
    Compute a perceptual hash (pHash) for a BGR image frame.

    Steps:
      1. Resize to (hash_size * highfreq_factor) square (32x32 by default)
      2. Convert to grayscale float64
      3. Apply 2D DCT (Discrete Cosine Transform)
      4. Extract top-left hash_size x hash_size block (low frequencies)
      5. Compute median of the block
      6. Return 64-bit integer hash (bits set where value > median)

    Args:
        frame_bgr: Input frame in BGR format (as returned by cv2.imread/VideoCapture)
        hash_size: Size of the final hash grid (8 = 64-bit hash)
        highfreq_factor: Multiplier for the resize dimension before DCT

    Returns:
        Integer hash value (64-bit for default hash_size=8)
    """
    img_size = hash_size * highfreq_factor  # 32x32

    # Resize and convert to grayscale
    resized = cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Apply 2D DCT
    dct_result = cv2.dct(gray)

    # Extract top-left low-frequency block
    dct_low = dct_result[:hash_size, :hash_size]

    # Compute median and threshold
    median_val = np.median(dct_low)
    hash_bits = (dct_low > median_val).flatten()

    # Convert boolean array to integer hash
    hash_int = 0
    for bit in hash_bits:
        hash_int = (hash_int << 1) | int(bit)

    return hash_int


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    Compute the Hamming distance between two integer hashes.
    This is the number of bit positions where the hashes differ.
    """
    return bin(hash1 ^ hash2).count('1')


def _format_timestamp(seconds: float) -> str:
    """Format seconds into a human-readable timestamp string (M:SS or H:MM:SS)."""
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def extract_keyframes(
    video_bytes: bytes,
    sample_fps: float = 2.0,
    hash_threshold: int = 10,
    max_dimension: int = 1024,
    jpeg_quality: int = 85,
) -> list[dict]:
    """
    Extract visually unique keyframes from a video.

    Args:
        video_bytes: Raw video file bytes
        sample_fps: How many frames per second to sample (before dedup)
        hash_threshold: Minimum Hamming distance between kept frames (0-64).
                        Higher = more aggressive deduplication.
        max_dimension: Maximum width or height for output frames
        jpeg_quality: JPEG compression quality (1-100)

    Returns:
        List of dicts, each containing:
          - "timestamp": Human-readable timestamp string (e.g., "0:05")
          - "base64": JPEG base64 data URI string
          - "frame_index": Original frame index in the video
    """
    # Write video bytes to a temp file (OpenCV requires filesystem access)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(tmp_fd, 'wb') as f:
            f.write(video_bytes)

        return _process_video_file(tmp_path, sample_fps, hash_threshold, max_dimension, jpeg_quality)
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _process_video_file(
    video_path: str,
    sample_fps: float,
    hash_threshold: int,
    max_dimension: int,
    jpeg_quality: int,
) -> list[dict]:
    """Internal: process a video file on disk and return keyframes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video file. The format may not be supported.")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0:
        log.warning("Could not determine video FPS, defaulting to 30")
        native_fps = 30.0

    if total_frames <= 0:
        raise ValueError("Video appears to have no frames.")

    duration_secs = total_frames / native_fps
    log.info(
        "Video opened: %.1f FPS, %d total frames, %.1f seconds duration",
        native_fps, total_frames, duration_secs
    )

    # Calculate which frame indices to sample
    frame_interval = max(1, int(native_fps / sample_fps))
    sample_indices = list(range(0, total_frames, frame_interval))
    log.info(
        "Sampling %d frames (every %d frames, ~%.1f FPS from %.1f native FPS)",
        len(sample_indices), frame_interval, sample_fps, native_fps
    )

    # Extract and deduplicate
    keyframes = []
    last_hash: Optional[int] = None
    frames_kept = 0
    frames_skipped = 0

    for target_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Compute perceptual hash
        current_hash = compute_phash(frame)

        # Deduplication check
        if last_hash is not None:
            dist = hamming_distance(current_hash, last_hash)
            if dist < hash_threshold:
                frames_skipped += 1
                continue

        # This frame is unique enough — keep it
        last_hash = current_hash
        frames_kept += 1

        # Calculate timestamp
        timestamp_secs = target_idx / native_fps
        timestamp_str = _format_timestamp(timestamp_secs)

        # Resize if needed (preserve aspect ratio)
        h, w = frame.shape[:2]
        if w > max_dimension or h > max_dimension:
            if w > h:
                new_w = max_dimension
                new_h = int(h * (max_dimension / w))
            else:
                new_h = max_dimension
                new_w = int(w * (max_dimension / h))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Encode as JPEG base64
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        success, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not success:
            log.warning("Failed to encode frame %d as JPEG, skipping", target_idx)
            continue

        b64_str = base64.b64encode(buffer.tobytes()).decode('ascii')
        data_uri = f"data:image/jpeg;base64,{b64_str}"

        keyframes.append({
            "timestamp": timestamp_str,
            "base64": data_uri,
            "frame_index": target_idx,
        })

    cap.release()

    log.info(
        "Keyframe extraction complete: %d kept, %d skipped (threshold=%d)",
        frames_kept, frames_skipped, hash_threshold
    )

    return keyframes
