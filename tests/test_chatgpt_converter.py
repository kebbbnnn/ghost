import unittest
from src.providers.chatgpt.converter import (
    parse_chatgpt_sse,
    chatgpt_event_to_openai_chunk,
    map_model,
    format_openai_chunk,
    build_non_streaming_response,
)


class TestChatGPTConverter(unittest.TestCase):
    def test_map_model(self):
        self.assertEqual(map_model("gpt-4o"), "auto")
        self.assertEqual(map_model("gpt-5-5"), "gpt-5-5")
        self.assertEqual(map_model("claude-3-5-sonnet-20241022"), "auto")
        self.assertEqual(map_model("unknown-model"), "auto")

    def test_parse_chatgpt_sse_standard(self):
        line = 'data: {"v": "hello"}\n\n'
        result = parse_chatgpt_sse(line)
        self.assertEqual(result, {"v": "hello"})

    def test_parse_chatgpt_sse_without_space(self):
        # W3C SSE standard permits "data:{...}" without space after colon
        line = 'data:{"v": "hello"}'
        result = parse_chatgpt_sse(line)
        self.assertEqual(result, {"v": "hello"})

    def test_parse_chatgpt_sse_done(self):
        self.assertEqual(parse_chatgpt_sse("data: [DONE]"), {"done": True})
        self.assertEqual(parse_chatgpt_sse("data:[DONE]"), {"done": True})

    def test_parse_chatgpt_sse_guards_non_dict_and_invalid(self):
        # Comments and empty lines
        self.assertIsNone(parse_chatgpt_sse(": heartbeat ping"))
        self.assertIsNone(parse_chatgpt_sse(""))
        self.assertIsNone(parse_chatgpt_sse("event: ping"))
        # Invalid JSON
        self.assertIsNone(parse_chatgpt_sse("data: {corrupted json"))
        # Non-dict JSON (e.g. data: "v1" from ChatGPT SSE header)
        self.assertIsNone(parse_chatgpt_sse('data: "v1"'))
        self.assertIsNone(parse_chatgpt_sse('data: 12345'))
        self.assertIsNone(parse_chatgpt_sse('data: ["item1"]'))

    def test_chatgpt_event_to_openai_chunk_done_and_complete(self):
        done_chunk = chatgpt_event_to_openai_chunk({"done": True}, "auto", "chunk-1")
        self.assertIsNotNone(done_chunk)
        self.assertEqual(done_chunk["choices"][0]["finish_reason"], "stop")
        self.assertEqual(done_chunk["choices"][0]["delta"], {})

        stream_complete_chunk = chatgpt_event_to_openai_chunk({"type": "message_stream_complete"}, "auto", "chunk-1")
        self.assertIsNotNone(stream_complete_chunk)
        self.assertEqual(stream_complete_chunk["choices"][0]["finish_reason"], "stop")

    def test_chatgpt_event_to_openai_chunk_delta_patch_list(self):
        # Case 1: list of operations
        event = {
            "v": [
                {"p": "/message/content/parts/0", "o": "append", "v": "Hello "},
                {"p": "/message/metadata/model", "o": "append", "v": "should_be_ignored"},
                {"p": "/message/content/parts/0", "o": "append", "v": "World<escape>!</escape>"},
            ]
        }
        chunk = chatgpt_event_to_openai_chunk(event, "auto", "chunk-1")
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk["choices"][0]["delta"]["content"], "Hello World!")
        self.assertIsNone(chunk["choices"][0]["finish_reason"])

    def test_chatgpt_event_to_openai_chunk_delta_patch_list_finish(self):
        event = {
            "v": [
                {"p": "/message/end_turn", "v": True}
            ]
        }
        chunk = chatgpt_event_to_openai_chunk(event, "auto", "chunk-1")
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk["choices"][0]["finish_reason"], "stop")

    def test_chatgpt_event_to_openai_chunk_case2_content_parts(self):
        # Case 2: String delta for content parts
        event_part0 = {"p": "/message/content/parts/0", "o": "append", "v": "Hello"}
        chunk0 = chatgpt_event_to_openai_chunk(event_part0, "auto", "chunk-1")
        self.assertIsNotNone(chunk0)
        self.assertEqual(chunk0["choices"][0]["delta"]["content"], "Hello")
        self.assertEqual(chunk0["choices"][0]["delta"]["role"], "assistant")

        # Shorthand continuation without "p"
        event_cont = {"v": " there"}
        chunk_cont = chatgpt_event_to_openai_chunk(event_cont, "auto", "chunk-1")
        self.assertIsNotNone(chunk_cont)
        self.assertEqual(chunk_cont["choices"][0]["delta"]["content"], " there")
        self.assertNotIn("role", chunk_cont["choices"][0]["delta"])

    def test_chatgpt_event_to_openai_chunk_case2_rejects_metadata(self):
        # Bug 59 Issue 1: metadata or status patch events must NEVER leak as content
        status_event = {"p": "/message/status", "o": "append", "v": "in_progress"}
        self.assertIsNone(chatgpt_event_to_openai_chunk(status_event, "auto", "chunk-1"))

        model_event = {"p": "/message/metadata/model_slug", "o": "append", "v": "gpt-4o"}
        self.assertIsNone(chatgpt_event_to_openai_chunk(model_event, "auto", "chunk-1"))

        weight_event = {"p": "/message/weight", "o": "append", "v": "1.0"}
        self.assertIsNone(chatgpt_event_to_openai_chunk(weight_event, "auto", "chunk-1"))

    def test_chatgpt_event_to_openai_chunk_end_turn_delta(self):
        end_turn_event = {"p": "/message/end_turn", "v": True}
        chunk = chatgpt_event_to_openai_chunk(end_turn_event, "auto", "chunk-1")
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunk["choices"][0]["delta"], {})

    def test_chatgpt_event_to_openai_chunk_legacy_multipart(self):
        # Bug 59 Issue 2: legacy format with multiple parts must concatenate all parts
        legacy_event = {
            "message": {
                "id": "msg-123",
                "author": {"role": "assistant"},
                "content": {
                    "parts": [
                        "First part. ",
                        "Second part with <escape>tag</escape>. ",
                        12345, # Non-string part ignored
                        "Third part.",
                    ]
                },
                "status": "finished_successfully",
                "recipient": "all",
            }
        }
        chunk = chatgpt_event_to_openai_chunk(legacy_event, "auto", "chunk-1")
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk["id"], "msg-123")
        self.assertEqual(
            chunk["choices"][0]["delta"]["content"],
            "First part. Second part with tag. Third part."
        )
        self.assertEqual(chunk["choices"][0]["finish_reason"], "stop")

    def test_build_non_streaming_response(self):
        res = build_non_streaming_response("Test output", "auto", "stop")
        self.assertEqual(res["object"], "chat.completion")
        self.assertEqual(res["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(res["choices"][0]["message"]["content"], "Test output")
        self.assertEqual(res["choices"][0]["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
