from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from PIL import Image

from ..config import BackendConfig
from ..models import Block


class Backend(ABC):
    """One page image in -> ordered list of Blocks out (two-step: layout, then per-block recognition)."""

    name: str = ""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @abstractmethod
    async def extract_page(self, image: Image.Image, semaphore: asyncio.Semaphore) -> list[Block]: ...

    async def aclose(self) -> None:  # release network resources bound to the current event loop
        return None
