"""HTTP liveness, readiness, and Prometheus endpoints."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest


class HealthState:
    def __init__(self) -> None:
        self._live = threading.Event()
        self._ready = threading.Event()

    @property
    def live(self) -> bool:
        return self._live.is_set()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def set_live(self, value: bool) -> None:
        (self._live.set if value else self._live.clear)()

    def set_ready(self, value: bool) -> None:
        (self._ready.set if value else self._ready.clear)()


class HealthHttpServer:
    def __init__(
        self,
        host: str,
        port: int,
        state: HealthState,
        registry: CollectorRegistry,
    ):
        handler = self._handler(state, registry)
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="health-http", daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        if self._thread.is_alive():
            self._server.shutdown()
            self._thread.join(timeout=5)
        self._server.server_close()

    @staticmethod
    def _handler(state: HealthState, registry: CollectorRegistry) -> type[BaseHTTPRequestHandler]:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib handler interface
                if self.path == "/health/live":
                    self._send_health(HTTPStatus.OK if state.live else HTTPStatus.SERVICE_UNAVAILABLE, state.live)
                    return
                if self.path == "/health/ready":
                    self._send_health(HTTPStatus.OK if state.ready else HTTPStatus.SERVICE_UNAVAILABLE, state.ready)
                    return
                if self.path == "/metrics":
                    payload = generate_latest(registry)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def _send_health(self, status: HTTPStatus, healthy: bool) -> None:
                payload = json.dumps({"status": "ok" if healthy else "unavailable"}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return

        return Handler
