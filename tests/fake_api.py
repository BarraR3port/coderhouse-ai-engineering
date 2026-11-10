import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

type ScriptedResponse = tuple[int, dict]


@dataclass
class RecordedRequest:
    path: str
    headers: dict[str, str]
    body: dict


@dataclass
class FakeAPI:
    base_url: str
    requests: list[RecordedRequest] = field(default_factory=list)
    _responses: list[ScriptedResponse] = field(default_factory=list)

    def respond(self, *responses: ScriptedResponse) -> None:
        self._responses.extend(responses)

    def next_response(self) -> ScriptedResponse:
        if not self._responses:
            return 599, {"error": {"message": "el test no preparó más respuestas"}}
        return self._responses.pop(0)

    def sent(self, index: int = -1) -> dict:
        return self.requests[index].body

    @property
    def models_called(self) -> list[str]:
        return [request.body.get("model") for request in self.requests]


@contextmanager
def running_fake_api() -> Iterator[FakeAPI]:
    api: FakeAPI | None = None

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            api.requests.append(RecordedRequest(self.path, dict(self.headers), body))
            status, payload = api.next_response()
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    api = FakeAPI(base_url=f"http://127.0.0.1:{server.server_port}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield api
    finally:
        server.shutdown()
        server.server_close()
