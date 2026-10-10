from __future__ import annotations

from ..config import BackendConfig
from .base import Backend

_DEFAULT_URLS = {"mineru": "http://127.0.0.1:8003", "teleocr": "http://127.0.0.1:8004"}


def create_backend(name: str, config: BackendConfig | None = None, **kwargs) -> Backend:
    name = name.lower()
    if config is None:
        if name not in _DEFAULT_URLS:
            raise ValueError(f"unknown backend {name!r}; choose from {sorted(_DEFAULT_URLS)}")
        config = BackendConfig(base_url=_DEFAULT_URLS[name])
    if name == "mineru":
        from .mineru import MinerUBackend

        return MinerUBackend(config, **kwargs)
    if name == "teleocr":
        from .teleocr import TeleOCRBackend

        return TeleOCRBackend(config, **kwargs)
    raise ValueError(f"unknown backend {name!r}; choose from {sorted(_DEFAULT_URLS)}")


__all__ = ["Backend", "create_backend"]
