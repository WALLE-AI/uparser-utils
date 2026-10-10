"""MinerU2.5 backend: thin adapter over `mineru_vl_utils.MinerUClient` (http-client mode).

All layout parsing and post-processing (OTSL->HTML, equation fixes, list merge, ...) is done by
mineru-vl-utils itself, so results match the upstream reference implementation.
"""
from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from PIL import Image

from ..config import BackendConfig
from ..models import Block
from .base import Backend


_CORE = {"type", "bbox", "content", "angle", "merge_prev"}


class MinerUBackend(Backend):
    name = "mineru"

    def __init__(self, config: BackendConfig, **client_kwargs: Any) -> None:
        super().__init__(config)
        try:
            from mineru_vl_utils import MinerUClient
        except ImportError as e:  # pragma: no cover
            raise ImportError("MinerU backend needs: pip install 'docparse-sdk[mineru]'") from e
        # mineru-vl-utils logs every raw layout output at DEBUG; silence by default
        # (re-enable with `from loguru import logger; logger.enable("mineru_vl_utils")`)
        logger.disable("mineru_vl_utils")
        kwargs: dict[str, Any] = dict(
            backend="http-client",
            model_name=config.model,
            server_url=config.normalized_url(),
            server_headers=config.request_headers() or None,
            max_concurrency=config.concurrency,
            http_timeout=int(config.timeout),
            connect_timeout=int(config.connect_timeout),
            max_retries=config.max_retries,
            use_tqdm=False,
        )
        if config.prompts:
            from mineru_vl_utils.mineru_client import DEFAULT_PROMPTS

            kwargs["prompts"] = {**DEFAULT_PROMPTS, **config.prompts}
        kwargs.update(client_kwargs)
        self.client = MinerUClient(**kwargs)

    async def extract_page(self, image: Image.Image, semaphore: asyncio.Semaphore) -> list[Block]:
        result = await self.client.aio_two_step_extract(image, semaphore=semaphore)
        return [
            Block(type=b["type"], bbox=list(b["bbox"]), content=b.get("content"), angle=b.get("angle"),
                  merge_prev=bool(b.get("merge_prev", False)),
                  extra={k: v for k, v in b.items() if k not in _CORE})
            for b in result
        ]

    async def aclose(self) -> None:
        # drop connections bound to this event loop; keep the client reusable for the next loop
        await self.client.client.aclose_current_loop()
