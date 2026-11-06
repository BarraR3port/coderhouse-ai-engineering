import json
from collections.abc import AsyncIterator, Iterable

import httpx2

type SseEvent = tuple[str | None, dict | str]

SSE_HEADERS = {"content-type": "text/event-stream"}


class RecordingTransport(httpx2.MockTransport):
    def __init__(self, responses: Iterable[httpx2.Response | Exception]) -> None:
        self.requests: list[httpx2.Request] = []
        self._responses = iter(responses)
        super().__init__(self._respond)

    def _respond(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        outcome = next(self._responses)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def sent_json(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)


class InterruptedStream(httpx2.AsyncByteStream):
    def __init__(self, delivered: bytes, error: Exception) -> None:
        self._delivered = delivered
        self._error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self._delivered:
            yield self._delivered
        raise self._error


def http_client(transport: RecordingTransport) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=transport)


def json_response(
    status: int, body: dict, headers: dict[str, str] | None = None
) -> httpx2.Response:
    return httpx2.Response(status, json=body, headers=headers)


def sse_body(events: Iterable[SseEvent]) -> bytes:
    lines = []
    for event, data in events:
        if event is not None:
            lines.append(f"event: {event}")
        payload = data if isinstance(data, str) else json.dumps(data)
        lines.append(f"data: {payload}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


def sse_response(events: Iterable[SseEvent]) -> httpx2.Response:
    return httpx2.Response(200, content=sse_body(events), headers=SSE_HEADERS)


def interrupted_sse(delivered: Iterable[SseEvent], error: Exception) -> httpx2.Response:
    return httpx2.Response(
        200, stream=InterruptedStream(sse_body(delivered), error), headers=SSE_HEADERS
    )


def openai_completion(content: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 30, "total_tokens": 42},
    }


def openai_stream_events(tokens: list[str]) -> list[SseEvent]:
    def chunk(delta: dict, finish_reason: str | None = None) -> dict:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    events: list[SseEvent] = [(None, chunk({"role": "assistant"}))]
    events += [(None, chunk({"content": token})) for token in tokens]
    events += [(None, chunk({}, "stop")), (None, "[DONE]")]
    return events


def openai_stream(tokens: list[str]) -> httpx2.Response:
    return sse_response(openai_stream_events(tokens))


def openai_error(status: int, code: str, headers: dict[str, str] | None = None) -> httpx2.Response:
    return json_response(
        status,
        {"error": {"message": f"simulated {code}", "type": code, "code": code}},
        headers,
    )


def anthropic_message(text: str) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": text},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 15, "output_tokens": 40},
    }


def anthropic_stream_events(tokens: list[str]) -> list[SseEvent]:
    events: list[SseEvent] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 15, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    ]
    events += [
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": token},
            },
        )
        for token in tokens
    ]
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": len(tokens)},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return events


def anthropic_stream(tokens: list[str]) -> httpx2.Response:
    return sse_response(anthropic_stream_events(tokens))


def anthropic_error_body(error_type: str) -> dict:
    return {"type": "error", "error": {"type": error_type, "message": f"simulated {error_type}"}}


def anthropic_error(status: int, error_type: str) -> httpx2.Response:
    return json_response(status, anthropic_error_body(error_type))


def anthropic_stream_error(error_type: str) -> httpx2.Response:
    return sse_response([("error", anthropic_error_body(error_type))])
