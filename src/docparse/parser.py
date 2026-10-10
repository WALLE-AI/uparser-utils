from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Sequence

from loguru import logger
from PIL import Image

from .backends import Backend, create_backend
from .config import BackendConfig
from . import official
from .assets import extract_assets
from .models import Document, Page
from .pdf import load_pages


class AsyncDocParser:
    """Async parser. Use as `async with AsyncDocParser(...) as p: doc = await p.parse(...)`."""

    def __init__(self, backend: str | Backend = "teleocr", *, base_url: str | None = None,
                 model: str | None = None, config: BackendConfig | None = None,
                 concurrency: int | None = None, dpi: int = 150, crop_assets: bool = True,
                 title_classifier=official.default_title_classifier, **backend_kwargs) -> None:
        if isinstance(backend, Backend):
            self.backend = backend
        else:
            if config is None and base_url:
                config = BackendConfig(base_url=base_url, model=model)
            if config is not None and concurrency:
                config.concurrency = concurrency
            self.backend = create_backend(backend, config, **backend_kwargs)
        self.dpi = dpi
        self.title_classifier = title_classifier   # (raw_block, page_index) -> (doc_title|paragraph_title, level)
        self.crop_assets = crop_assets   # crop image/seal/char blocks so md/middle.json can reference them
        self._sem: asyncio.Semaphore | None = None

    async def __aenter__(self) -> "AsyncDocParser":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.backend.aclose()
        self._sem = None

    def _semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.backend.config.concurrency)
        return self._sem

    async def parse(self, source, *, pages: Sequence[int] | None = None) -> Document:
        loaded = load_pages(source, dpi=self.dpi, pages=pages)
        sem = self._semaphore()

        async def one(idx: int, img: Image.Image, size: tuple[int, int]) -> Page:
            blocks = await self.backend.extract_page(img, sem)
            for b in blocks:
                b.page = idx
            assets = extract_assets(img, blocks) if self.crop_assets else {}
            raw = None
            if self.crop_assets and official.available():
                raw = official.build_raw_page(img, blocks, self.backend.name, idx, self.title_classifier)
            return Page(index=idx, width=img.width, height=img.height, page_size=size, blocks=blocks,
                        assets=assets, raw=raw)

        result = await asyncio.gather(*(one(i, img, sz) for i, img, sz in loaded))
        name = str(source) if isinstance(source, (str, Path)) else ""
        logger.info("parsed {} page(s) with {}", len(result), self.backend.name)
        partial = pages is not None and not isinstance(source, Image.Image)
        return Document(pages=list(result), backend=self.backend.name, source=name, partial=partial)

    async def parse_many(self, sources: Sequence, **kw) -> list[Document]:
        return list(await asyncio.gather(*(self.parse(s, **kw) for s in sources)))


class DocParser:
    """Blocking facade over AsyncDocParser (each call runs its own event loop; do not call from async code)."""

    def __init__(self, *args, **kwargs) -> None:
        self._inner = AsyncDocParser(*args, **kwargs)

    def _run(self, coro_fn, *a, **kw):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("DocParser is blocking and cannot run inside an event loop; use AsyncDocParser")

        async def main():
            try:
                return await coro_fn(*a, **kw)
            finally:
                await self._inner.aclose()  # connections are bound to this loop

        return asyncio.run(main())

    def parse(self, source, *, pages: Sequence[int] | None = None) -> Document:
        return self._run(self._inner.parse, source, pages=pages)

    def parse_many(self, sources: Sequence, **kw) -> list[Document]:
        return self._run(self._inner.parse_many, sources, **kw)
