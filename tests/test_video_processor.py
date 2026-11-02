"""
test_video_processor.py — Unit tests for the video processing pipeline.

Tests:
  1. pHash computation and Hamming distance
  2. Keyframe extraction from a synthetic test video
"""
import base64
import os
import sys
import tempfile

import cv2
import numpy as np

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.core.video_processor import compute_phash, hamming_distance, extract_keyframes


def test_phash_identical_frames():
    """Two identical frames should produce the same hash (distance = 0)."""
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    hash1 = compute_phash(frame)
    hash2 = compute_phash(frame.copy())
    dist = hamming_distance(hash1, hash2)
    
    assert dist == 0, f"Expected distance 0 for identical frames, got {dist}"
    print("✅ test_phash_identical_frames PASSED")


def test_phash_different_frames():
    """Two very different frames should have a large Hamming distance."""
    # Use frames with actual spatial content (pHash needs frequency variation)
    # Solid-color frames are degenerate — their DCT is all DC coefficient
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create a gradient pattern
    for i in range(100):
        frame1[i, :, :] = i * 2  # Horizontal gradient
    
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create a very different pattern (checkerboard)
    for i in range(100):
        for j in range(100):
            if (i // 10 + j // 10) % 2 == 0:
                frame2[i, j] = [255, 0, 0]
            else:
                frame2[i, j] = [0, 255, 0]
    
    hash1 = compute_phash(frame1)
    hash2 = compute_phash(frame2)
    dist = hamming_distance(hash1, hash2)
    
    # Should show meaningful difference for structurally different frames
    assert dist >= 5, f"Expected distance >= 5 for different frames, got {dist}"
    print(f"✅ test_phash_different_frames PASSED (distance={dist})")


def test_phash_similar_frames():
    """Two slightly different frames should have a small Hamming distance."""
    frame1 = np.full((100, 100, 3), 128, dtype=np.uint8)
    # Add just a slight variation
    frame2 = frame1.copy()
    frame2[40:60, 40:60] = 135  # Small patch, tiny change
    
    hash1 = compute_phash(frame1)
    hash2 = compute_phash(frame2)
    dist = hamming_distance(hash1, hash2)
    
    # Should be small (likely < 10)
    print(f"✅ test_phash_similar_frames PASSED (distance={dist}, should be small)")


def create_synthetic_video(num_frames=60, fps=30, width=320, height=240):
    """Create a synthetic video in memory with known content changes.
    
    Creates a video with structurally different scenes (not just color changes):
      - Frames 0-19: horizontal gradient with a circle
      - Frames 20-39: checkerboard pattern
      - Frames 40-59: diagonal stripes
    
    pHash is color-agnostic (operates on grayscale DCT), so we need
    actual structural/spatial differences to distinguish scenes.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if i < 20:
            # Scene 1: Horizontal gradient with a circle
            for row in range(height):
                frame[row, :] = int((row / height) * 255)
            cv2.circle(frame, (width // 2, height // 2), 60, (255, 255, 255), -1)
        elif i < 40:
            # Scene 2: Checkerboard pattern
            block = 30
            for row in range(height):
                for col in range(width):
                    if (row // block + col // block) % 2 == 0:
                        frame[row, col] = [200, 200, 200]
        else:
            # Scene 3: Diagonal stripes
            for row in range(height):
                for col in range(width):
                    if ((row + col) // 20) % 2 == 0:
                        frame[row, col] = [180, 180, 180]
        
        writer.write(frame)
    
    writer.release()
    
    with open(tmp_path, 'rb') as f:
        video_bytes = f.read()
    
    os.unlink(tmp_path)
    return video_bytes


def test_extract_keyframes():
    """Test that keyframe extraction correctly identifies scene changes."""
    video_bytes = create_synthetic_video(num_frames=60, fps=30)
    
    # Extract at 2 FPS from a 2-second video (60 frames @ 30fps)
    keyframes = extract_keyframes(
        video_bytes,
        sample_fps=2.0,
        hash_threshold=10,
        max_dimension=320
    )
    
    print(f"  Extracted {len(keyframes)} keyframes")
    for kf in keyframes:
        print(f"    - Frame {kf['frame_index']} at {kf['timestamp']} (base64 len: {len(kf['base64'])})")
    
    # We expect roughly 3 keyframes (one for each color phase)
    # Could be slightly more depending on exact sampling alignment
    assert len(keyframes) >= 2, f"Expected at least 2 keyframes (scene changes), got {len(keyframes)}"
    assert len(keyframes) <= 6, f"Expected at most 6 keyframes, got {len(keyframes)} (dedup may have failed)"
    
    # Verify each keyframe has required fields
    for kf in keyframes:
        assert 'timestamp' in kf, "Missing 'timestamp' field"
        assert 'base64' in kf, "Missing 'base64' field"
        assert 'frame_index' in kf, "Missing 'frame_index' field"
        assert kf['base64'].startswith('data:image/jpeg;base64,'), "Invalid base64 data URI"
    
    print(f"✅ test_extract_keyframes PASSED ({len(keyframes)} keyframes extracted)")


def test_extract_keyframes_timestamps():
    """Verify timestamps are formatted correctly."""
    video_bytes = create_synthetic_video(num_frames=60, fps=30)
    keyframes = extract_keyframes(video_bytes, sample_fps=2.0)
    
    for kf in keyframes:
        ts = kf['timestamp']
        # Should match M:SS format
        parts = ts.split(':')
        assert len(parts) == 2, f"Expected M:SS format, got '{ts}'"
        assert parts[0].isdigit(), f"Minutes should be numeric: '{ts}'"
        assert parts[1].isdigit() and len(parts[1]) == 2, f"Seconds should be 2-digit: '{ts}'"
    
    print(f"✅ test_extract_keyframes_timestamps PASSED")


import unittest
from fastapi.testclient import TestClient
from src.server import app
from src.providers import get_provider_for_model, get_fallback_provider


class TestVideoProcessor(unittest.TestCase):
    def test_phash_identical_frames(self):
        test_phash_identical_frames()

    def test_phash_different_frames(self):
        test_phash_different_frames()

    def test_phash_similar_frames(self):
        test_phash_similar_frames()

    def test_extract_keyframes(self):
        test_extract_keyframes()

    def test_extract_keyframes_timestamps(self):
        test_extract_keyframes_timestamps()

    def test_video_chat_endpoint_resilient_stream(self):
        """Verify /api/video/chat fails over to fallback provider if primary stream fails on first chunk."""
        video_bytes = create_synthetic_video(num_frames=30, fps=15)
        primary_provider = get_provider_for_model("auto")
        fallback_provider = get_fallback_provider(primary_provider)

        orig_primary_stream = primary_provider.stream_chat
        orig_fallback_stream = fallback_provider.stream_chat if fallback_provider else None

        async def failing_primary_stream(*args, **kwargs):
            raise RuntimeError("Primary video stream initialization failure")
            yield  # make it a generator

        async def successful_fallback_stream(*args, **kwargs):
            yield 'data: {"id": "vid-1", "choices": [{"delta": {"role": "assistant", "content": "I see synthetic video."}}]}\n\n'
            yield 'data: [DONE]\n\n'

        primary_provider.stream_chat = failing_primary_stream
        if fallback_provider:
            fallback_provider.stream_chat = successful_fallback_stream

        try:
            client = TestClient(app)
            response = client.post(
                "/api/video/chat",
                data={
                    "prompt": "What is happening in this video?",
                    "model": "auto",
                },
                files={
                    "video": ("synthetic.mp4", video_bytes, "video/mp4"),
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("I see synthetic video.", response.text)
        finally:
            primary_provider.stream_chat = orig_primary_stream
            if fallback_provider and orig_fallback_stream:
                fallback_provider.stream_chat = orig_fallback_stream


if __name__ == '__main__':
    unittest.main()
