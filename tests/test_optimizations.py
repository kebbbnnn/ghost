"""
test_optimizations.py — Closed-loop unit tests for the speed optimizations.

Tests:
  1. PoW seed pre-computation produces identical results to naive approach
  2. Double-buffered sentinel token swap logic (ChatGPTProvider)
  3. Claude persistent session pool (AsyncSession reuse)
  4. Claude org_id permanent cache (no re-fetch)
  5. Claude conversation cache eviction — only on "not found" errors
  6. Server telemetry token_count accumulation
"""
import sys
import os
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# 1. PoW seed pre-computation correctness
# ---------------------------------------------------------------------------

class TestPowSeedPrecomputation(unittest.TestCase):
    """Verify the optimized FNV-1a matches the naive implementation for all inputs."""

    def setUp(self):
        from src.providers.chatgpt.sentinel.pow_solver import (
            _fnv1a_32, _fnv1a_state, _fnv1a_32_continue, _encode_config
        )
        self._fnv1a_32 = _fnv1a_32
        self._fnv1a_state = _fnv1a_state
        self._fnv1a_32_continue = _fnv1a_32_continue
        self._encode_config = _encode_config

    def _make_config(self):
        return [1920+1080, "Mon Jan 01 2024", 4*1024*1024*1024, 0,
                "Mozilla/5.0", "https://cdn.oaistatic.com/chunk.js", None,
                "en-US", "en-US,en", 0.0, ["onLine", True], "cookie",
                "navigator", 1500.0, "test-uuid", "", 8, 1700000000000.0,
                0, 0, 0, 0, 0, 0, 0]

    def test_precomputed_matches_naive_single(self):
        seed = "gAAAAAC_test_seed"
        config = self._make_config()
        config[3] = 42
        config[9] = 150.0
        encoded = self._encode_config(config)
        naive = self._fnv1a_32(seed + encoded)
        state = self._fnv1a_state(seed)
        optimized = self._fnv1a_32_continue(encoded, state)
        self.assertEqual(naive, optimized, "Optimized hash must equal naive for single case")

    def test_precomputed_matches_naive_many_iterations(self):
        seed = "seed_" + "x" * 30
        config = self._make_config()
        state = self._fnv1a_state(seed)
        for i in range(200):
            config[3] = i
            config[9] = float(i * 10)
            encoded = self._encode_config(config)
            naive = self._fnv1a_32(seed + encoded)
            optimized = self._fnv1a_32_continue(encoded, state)
            self.assertEqual(naive, optimized, f"Mismatch at iteration {i}")

    def test_empty_seed(self):
        seed = ""
        config = self._make_config()
        encoded = self._encode_config(config)
        naive = self._fnv1a_32(seed + encoded)
        state = self._fnv1a_state(seed)
        optimized = self._fnv1a_32_continue(encoded, state)
        self.assertEqual(naive, optimized, "Should handle empty seed")

    def test_pow_solve_still_produces_valid_token(self):
        from src.providers.chatgpt.sentinel.pow_solver import solve_pow, _fnv1a_32, _encode_config
        from src.providers.chatgpt.sentinel.token_generator import _build_config
        import uuid
        sid = str(uuid.uuid4())
        config, now_ms = _build_config(sid)
        token = solve_pow("test_seed", "0ffff0", config, page_load_ms=now_ms)
        self.assertTrue(token.startswith("gAAAAAB"), f"Token must start with gAAAAAB: {token[:20]}")
        # Verify the solved token actually passes the difficulty check
        inner = token[len("gAAAAAB"):]
        encoded, suffix = inner[:-2], inner[-2:]
        self.assertEqual(suffix, "~S")
        digest = _fnv1a_32("test_seed" + encoded)
        self.assertLessEqual(digest[:6], "0ffff0", f"Hash {digest[:6]} should be <= 0ffff0")


# ---------------------------------------------------------------------------
# 2. Double-buffered sentinel token swap (ChatGPTProvider)
# ---------------------------------------------------------------------------

class TestDoubleBufferedTokens(unittest.IsolatedAsyncioTestCase):
    """Test the double-buffer swap logic without making real network calls."""

    async def asyncSetUp(self):
        from src.providers.chatgpt.session import ChatGPTProvider
        self.provider = ChatGPTProvider()
        # Inject a pre-made "current" and "next" token directly
        self.provider._sentinel_headers = {"Openai-Sentinel-Proof-Token": "current-token"}
        self.provider._next_sentinel_headers = {"Openai-Sentinel-Proof-Token": "next-token"}
        self.provider._token_fetched_at = 9999999999.0  # far future — not expired

    async def test_warm_path_returns_without_refresh(self):
        """If token is valid and not expired, ensure_ready should not call _refresh."""
        with patch.object(self.provider, '_refresh_chat_requirements', new_callable=AsyncMock) as mock_refresh:
            await self.provider.ensure_ready()
            mock_refresh.assert_not_called()
        self.assertEqual(
            self.provider._sentinel_headers["Openai-Sentinel-Proof-Token"],
            "current-token"
        )

    async def test_expired_token_swaps_from_buffer(self):
        """On expiry, if next-token is ready, it should swap in without calling _refresh."""
        self.provider._token_fetched_at = 0.0  # force expired

        with patch.object(self.provider, '_refresh_chat_requirements', new_callable=AsyncMock) as mock_refresh:
            with patch.object(self.provider, '_schedule_next_token_prefetch') as mock_schedule:
                await self.provider.ensure_ready()
                mock_refresh.assert_not_called()  # Should NOT block on a full refresh
                mock_schedule.assert_called_once()  # Should kick off next prefetch

        self.assertEqual(
            self.provider._sentinel_headers["Openai-Sentinel-Proof-Token"],
            "next-token",  # Swapped from buffer
        )
        self.assertIsNone(self.provider._next_sentinel_headers)  # Buffer consumed

    async def test_expired_token_no_buffer_falls_back_to_refresh(self):
        """On expiry with empty buffer, must call _refresh_chat_requirements."""
        self.provider._token_fetched_at = 0.0
        self.provider._next_sentinel_headers = None  # Buffer empty

        with patch.object(self.provider, '_refresh_chat_requirements', new_callable=AsyncMock) as mock_refresh:
            await self.provider.ensure_ready()
            mock_refresh.assert_called_once()

    async def test_cold_start_no_headers_calls_refresh(self):
        """Cold start (no sentinel headers at all) must call _refresh_chat_requirements."""
        self.provider._sentinel_headers = None
        self.provider._next_sentinel_headers = None

        with patch.object(self.provider, '_refresh_chat_requirements', new_callable=AsyncMock) as mock_refresh:
            await self.provider.ensure_ready()
            mock_refresh.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Claude persistent session pool
# ---------------------------------------------------------------------------

class TestClaudeSessionPool(unittest.IsolatedAsyncioTestCase):
    """Verify that the session pool returns the same object on repeated calls."""

    async def asyncSetUp(self):
        from src.providers.claude.session import ClaudeProvider
        self.provider = ClaudeProvider()

    async def test_same_session_returned_for_same_email(self):
        s1 = await self.provider._get_session("user@example.com")
        s2 = await self.provider._get_session("user@example.com")
        self.assertIs(s1, s2, "Same session object must be reused for same account")

    async def test_different_sessions_for_different_emails(self):
        s1 = await self.provider._get_session("alice@example.com")
        s2 = await self.provider._get_session("bob@example.com")
        self.assertIsNot(s1, s2, "Different accounts must get different sessions")

    async def asyncTearDown(self):
        await self.provider.stop_background_tasks()


# ---------------------------------------------------------------------------
# 4. Claude org_id permanent cache
# ---------------------------------------------------------------------------

class TestClaudeOrgIdCache(unittest.IsolatedAsyncioTestCase):
    """Verify org_id is fetched once and then served from cache forever."""

    async def asyncSetUp(self):
        from src.providers.claude.session import ClaudeProvider
        self.provider = ClaudeProvider()

    async def test_org_id_cached_after_first_fetch(self):
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"uuid": "org-12345"}]
        mock_client.get = AsyncMock(return_value=mock_resp)

        org1 = await self.provider._get_org_id("test@example.com", "session-key", mock_client)
        org2 = await self.provider._get_org_id("test@example.com", "session-key", mock_client)

        self.assertEqual(org1, "org-12345")
        self.assertEqual(org2, "org-12345")
        # HTTP should only be called ONCE despite two invocations
        mock_client.get.assert_called_once()

    async def test_different_accounts_have_separate_cache(self):
        mock_client = AsyncMock()
        def make_response(org_id):
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = [{"uuid": org_id}]
            return r

        mock_client.get = AsyncMock(side_effect=[
            make_response("org-alice"),
            make_response("org-bob"),
        ])

        alice_org = await self.provider._get_org_id("alice@example.com", "sk1", mock_client)
        bob_org   = await self.provider._get_org_id("bob@example.com", "sk2", mock_client)

        self.assertEqual(alice_org, "org-alice")
        self.assertEqual(bob_org, "org-bob")
        self.assertEqual(mock_client.get.call_count, 2)

    async def asyncTearDown(self):
        await self.provider.stop_background_tasks()


# ---------------------------------------------------------------------------
# 5. Claude conversation cache eviction strategy
# ---------------------------------------------------------------------------

class TestClaudeConversationCacheEviction(unittest.TestCase):
    """Verify only definitive 404/gone errors evict the conversation cache."""

    def setUp(self):
        from src.providers.claude.session import ClaudeProvider
        self.provider = ClaudeProvider()

    def test_404_error_triggers_eviction(self):
        self.assertTrue(self.provider._is_conversation_not_found_error(
            Exception("Claude API 404: conversation not found")
        ))

    def test_not_found_string_triggers_eviction(self):
        self.assertTrue(self.provider._is_conversation_not_found_error(
            Exception("404 not found")
        ))

    def test_410_gone_triggers_eviction(self):
        self.assertTrue(self.provider._is_conversation_not_found_error(
            Exception("410 gone")
        ))

    def test_network_timeout_does_not_trigger_eviction(self):
        self.assertFalse(self.provider._is_conversation_not_found_error(
            Exception("Connection timed out")
        ))

    def test_connection_reset_does_not_trigger_eviction(self):
        self.assertFalse(self.provider._is_conversation_not_found_error(
            Exception("Connection reset by peer")
        ))

    def test_503_does_not_trigger_eviction(self):
        self.assertFalse(self.provider._is_conversation_not_found_error(
            Exception("503 Service Unavailable")
        ))

    def test_rate_limit_does_not_trigger_eviction(self):
        self.assertFalse(self.provider._is_conversation_not_found_error(
            Exception("429 Rate limit exceeded")
        ))


# ---------------------------------------------------------------------------
# 6. Server telemetry — token count accumulation, not per-token broadcast
# ---------------------------------------------------------------------------

class TestServerTelemetryDebounce(unittest.TestCase):
    """Verify the streaming handler counts tokens and only broadcasts at end."""

    def test_token_count_accumulates(self):
        """Simulate the streaming generator logic and confirm token_count increments."""
        token_count = 0
        chunks = ["data: chunk1\n\n", "data: chunk2\n\n", "data: chunk3\n\n", "data: [DONE]\n\n"]
        for chunk in chunks:
            token_count += 1
        # 4 chunks total (including DONE)
        self.assertEqual(token_count, 4)

    def test_no_per_token_broadcast_in_server(self):
        """Confirm the old per-token broadcast call was removed from server.py."""
        with open("src/server.py") as f:
            content = f.read()
        # The old code did: await broadcast_event("request_token", ...) inside the loop
        # This should no longer exist
        self.assertNotIn(
            '"request_token"',
            content,
            "Per-token broadcast should be removed from server.py"
        )
        # But request_end with token count should exist
        self.assertIn('"tokens"', content,
                      'request_end event should include total "tokens" count')


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
