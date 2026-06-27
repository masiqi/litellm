"""
Tests for AnthropicResponsesStreamWrapper
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py)
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
)

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)


def _process_all(events: list) -> list:
    wrapper = AnthropicResponsesStreamWrapper(responses_stream=None, model="m")
    for event in events:
        wrapper._process_event(event)
    return list(wrapper._chunk_queue)


class _AsyncEventStream:
    def __init__(self, events: list) -> None:
        self._events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration


class TestProcessEventTextDeltaWithoutOutputItemAdded:
    """Streams that skip response.output_item.added (e.g. LMStudio) must still
    open a text block before any delta and never emit index -1."""

    def test_process_event_synthesizes_content_block_start_before_delta(self):
        chunks = _process_all(
            [
                {"type": "response.output_text.delta", "item_id": "i1", "delta": "Hel"},
                {"type": "response.output_text.delta", "item_id": "i1", "delta": "lo"},
            ]
        )
        assert [c["type"] for c in chunks] == [
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
        ]
        assert chunks[0]["content_block"] == {"type": "text", "text": ""}
        assert [c["index"] for c in chunks] == [0, 0, 0]
        assert chunks[1]["delta"] == {"type": "text_delta", "text": "Hel"}

    def test_process_event_delta_without_item_id_never_yields_negative_index(self):
        chunks = _process_all([{"type": "response.output_text.delta", "delta": "Hi"}])
        assert [(c["type"], c["index"]) for c in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
        ]

    def test_process_event_unregistered_item_id_opens_new_text_block(self):
        chunks = _process_all(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "reasoning", "id": "rs_1"},
                },
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Hi"},
            ]
        )
        assert chunks[1]["type"] == "content_block_start"
        assert chunks[1]["content_block"] == {"type": "text", "text": ""}
        assert [c["index"] for c in chunks[1:]] == [1, 1]

    def test_process_event_registered_item_id_does_not_synthesize_start(self):
        chunks = _process_all(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "message", "id": "m1"},
                },
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Hi"},
            ]
        )
        assert [(c["type"], c["index"]) for c in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
        ]


class TestAnthropicResponsesStreamWrapperAsyncIteration:
    @pytest.mark.asyncio
    async def test_does_not_duplicate_message_start_when_response_created_arrives(self):
        wrapper = AnthropicResponsesStreamWrapper(
            responses_stream=_AsyncEventStream(
                [
                    {"type": "response.created"},
                    {
                        "type": "response.output_item.added",
                        "item": {"type": "message", "id": "msg_1"},
                    },
                ]
            ),
            model="m",
        )

        chunks = [await wrapper.__anext__(), await wrapper.__anext__()]

        assert [chunk["type"] for chunk in chunks] == [
            "message_start",
            "content_block_start",
        ]

    @pytest.mark.asyncio
    async def test_sends_fallback_message_start_before_first_content_block(self):
        wrapper = AnthropicResponsesStreamWrapper(
            responses_stream=_AsyncEventStream(
                [
                    {
                        "type": "response.output_item.added",
                        "item": {"type": "message", "id": "msg_1"},
                    },
                ]
            ),
            model="m",
        )

        chunks = [await wrapper.__anext__(), await wrapper.__anext__()]

        assert [chunk["type"] for chunk in chunks] == [
            "message_start",
            "content_block_start",
        ]

    def test_reasoning_delta_opens_thinking_block_for_unregistered_item(self):
        chunks = _process_all(
            [
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": "rs_1",
                    "delta": "Need to call Bash.",
                }
            ]
        )

        assert [c["type"] for c in chunks] == [
            "content_block_start",
            "content_block_delta",
        ]
        assert chunks[0]["content_block"] == {"type": "thinking", "thinking": ""}
        assert chunks[1]["index"] == 0
        assert chunks[1]["delta"] == {
            "type": "thinking_delta",
            "thinking": "Need to call Bash.",
        }
