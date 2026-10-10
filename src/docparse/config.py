from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BackendConfig:
    base_url: str                      # e.g. http://127.0.0.1:8004  (a trailing /v1 is tolerated)
    model: str | None = None           # served model name; None -> autodetect via /v1/models
    api_key: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    concurrency: int = 16              # max in-flight requests to the server
    timeout: float = 600.0
    connect_timeout: float = 10.0
    max_retries: int = 3               # transport errors / 429 / 5xx, exponential backoff (0.5s, 1s, 2s ...)
    # Per-task prompt overrides, e.g. {"layout": "...", "table": "..."}. None keeps backend defaults.
    prompts: dict[str, str] | None = None

    def normalized_url(self) -> str:
        url = self.base_url.rstrip("/")
        return url[:-3] if url.endswith("/v1") else url

    def request_headers(self) -> dict[str, str]:
        h = dict(self.headers)
        if self.api_key:
            h.setdefault("Authorization", f"Bearer {self.api_key}")
        return h
