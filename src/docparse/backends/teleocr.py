"""TeleOCR backend.

TeleOCR's reference client has no HTTP mode, so the transport is implemented here with httpx.
Prompts, layout parsing, crop preparation and the post-processing pipeline are ported from
TeleOCR/vlm_utils/TeleOCR_client.py (post-processing itself is vendored unchanged).
"""
from __future__ import annotations

import asyncio
import base64
import io
import math
import re
from typing import Any, Literal

import httpx
from loguru import logger
from PIL import Image, ImageDraw

from .._vendor.teleocr.post_process import post_process
from .._vendor.teleocr.structs import BLOCK_TYPES, ContentBlock
from ..config import BackendConfig
from ..models import Block
from .base import Backend

LAYOUT_PROMPTS = {
    "detection": "\nAnalyze the image layout.",
    "segmentation": "\nMulti-point Layout Segmentation Analysis.",
}
DEFAULT_PROMPTS: dict[str, str] = {
    "text": "\nPlease output the text content from the image.",
    "table": "\nThis is the image of a table. Please output the table in OTSL format.",
    "equation": "\nPlease write out the expression of the formula in the image using LaTeX format.",
    "code": "\nThe image contains a code snippet, please output the parsing result.",
    "seal": "\nSeal Recognition:",
    "char": "\nThis is a scientific figure. Please extract the table implied by this figure.",
    "default": "\nPlease output the text content from the image.",
}
# (presence_penalty, frequency_penalty) per block type, from upstream DEFAULT_SAMPLING_PARAMS
_PENALTIES = {
    "table": (1.0, 0.005), "char": (1.0, 0.005), "seal": (1.0, 0.005),
    "layout": (0.0, 0.0),
}
_DEFAULT_PENALTY = (1.0, 0.05)
_RETRY_STATUS = {429, 500, 502, 503, 504}
_SKIP_EXTRACT = {"image", "list", "equation_block"}
_ANGLES: dict[str, Literal[0, 90, 180, 270]] = {"up": 0, "right": 90, "down": 180, "left": 270}
_LAYOUT_RE = re.compile(r"^<box:([\d\s]+)><label:(\w+)><([^>]+)>$")


class _RetryableStatus(Exception):
    pass


def parse_layout_output(output: str) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _LAYOUT_RE.match(line)
        if not m:
            logger.warning("TeleOCR layout line does not match expected format: {}", line)
            continue
        nums_s, ref_type, tail = m.groups()
        ref_type = ref_type.lower()
        if ref_type not in BLOCK_TYPES:
            logger.warning("unknown TeleOCR block type: {}", ref_type)
            continue
        nums = [int(v) for v in nums_s.split()]
        if len(nums) < 4 or len(nums) % 2 or any(c < 0 or c > 1000 for c in nums):
            logger.warning("invalid TeleOCR box: {}", nums_s)
            continue
        xs, ys = nums[0::2], nums[1::2]
        if min(xs) == max(xs) or min(ys) == max(ys):          # zero-area box (mineru-vl-utils rejects these too)
            logger.warning("degenerate TeleOCR box dropped: {}", nums_s)
            continue
        angle = next((a for k, a in _ANGLES.items() if k in tail), None)
        blocks.append(ContentBlock(ref_type, [c / 1000.0 for c in nums], angle=angle))
    return blocks


def _resize_by_need(image: Image.Image, min_edge: int = 28, max_ratio: float = 50) -> Image.Image:
    w, h = image.size
    if max(w, h) / min(w, h) > max_ratio:
        nw, nh = (w, math.ceil(w / max_ratio)) if w > h else (math.ceil(h / max_ratio), h)
        canvas = Image.new(image.mode, (nw, nh), (255, 255, 255))
        canvas.paste(image, ((nw - w) // 2, (nh - h) // 2))
        image = canvas
    if min(image.size) < min_edge:
        s = min_edge / min(image.size)
        image = image.resize((math.ceil(image.width * s), math.ceil(image.height * s)), Image.Resampling.BICUBIC)
    return image


def crop_block(image: Image.Image, block: ContentBlock) -> Image.Image | None:
    """Crop a rectangle (4 coords) or polygon (>4 coords, outside area whitened-out to black like upstream)."""
    W, H = image.size
    pts = [(int(block.bbox[i] * W), int(block.bbox[i + 1] * H)) for i in range(0, len(block.bbox), 2)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    crop = image.crop((x1, y1, x2, y2))
    if len(pts) > 2:
        mask = Image.new("L", crop.size, 0)
        ImageDraw.Draw(mask).polygon([(x - x1, y - y1) for x, y in pts], fill=255)
        bg = Image.new("RGB", crop.size, (0, 0, 0))
        crop = Image.composite(crop, bg, mask)
    if block.angle in (90, 180, 270):
        crop = crop.rotate(block.angle, expand=True)
    return _resize_by_need(crop)


def _data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class TeleOCRBackend(Backend):
    name = "teleocr"

    def __init__(
        self,
        config: BackendConfig,
        *,
        layout_mode: Literal["detection", "segmentation"] = "detection",
        layout_image_size: tuple[int, int] = (1036, 1036),
        max_tokens: int = 4096,
        no_repeat_ngram_size: int | None = None,   # upstream uses 100 via a server-side logits processor
        abandon_list: bool = False,
        abandon_paratext: bool = False,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        super().__init__(config)
        self.prompts = {**DEFAULT_PROMPTS, **(config.prompts or {})}
        self.layout_prompt = self.prompts.get("layout") or LAYOUT_PROMPTS[layout_mode]
        self.layout_image_size = layout_image_size
        self.max_tokens = max_tokens
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.abandon_list = abandon_list
        self.abandon_paratext = abandon_paratext
        self.system_prompt = system_prompt
        self._client: httpx.AsyncClient | None = None
        self._model: str | None = config.model

    # ---- transport ----------------------------------------------------------------------
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            # keepalive_expiry < uvicorn's 5s keep-alive: avoid reusing a connection the server just closed
            transport = httpx.AsyncHTTPTransport(
                retries=self.config.max_retries, limits=httpx.Limits(keepalive_expiry=2.0, max_connections=None))
            self._client = httpx.AsyncClient(
                base_url=self.config.normalized_url(),
                headers=self.config.request_headers(),
                timeout=httpx.Timeout(self.config.timeout, connect=self.config.connect_timeout),
                transport=transport,
            )
        return self._client

    async def _model_name(self) -> str:
        if self._model is None:
            r = await self._http().get("/v1/models")
            r.raise_for_status()
            self._model = r.json()["data"][0]["id"]
        return self._model

    async def _chat(self, image: Image.Image, prompt: str, task: str, semaphore: asyncio.Semaphore) -> str:
        presence, frequency = _PENALTIES.get(task, _DEFAULT_PENALTY)
        payload: dict[str, Any] = {
            "model": await self._model_name(),
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": _data_uri(image)}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            "temperature": 0.0, "top_p": 0.01, "top_k": 1,
            "presence_penalty": presence, "frequency_penalty": frequency, "repetition_penalty": 1.0,
            "max_tokens": self.max_tokens,
            "skip_special_tokens": False,
        }
        if self.no_repeat_ngram_size:
            payload["vllm_xargs"] = {"no_repeat_ngram_size": self.no_repeat_ngram_size}
        retries = self.config.max_retries
        for attempt in range(retries + 1):
            try:
                async with semaphore:                      # released while backing off
                    r = await self._http().post("/v1/chat/completions", json=payload)
                if r.status_code in _RETRY_STATUS and attempt < retries:
                    raise _RetryableStatus(r.status_code)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except (httpx.TransportError, _RetryableStatus) as e:
                if attempt >= retries:
                    raise
                delay = min(0.5 * 2 ** attempt, 8.0)
                logger.warning("TeleOCR request failed ({}); retry {}/{} in {:.1f}s", type(e).__name__ if not
                               isinstance(e, _RetryableStatus) else f"HTTP {e}", attempt + 1, retries, delay)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    # ---- pipeline -----------------------------------------------------------------------
    async def extract_page(self, image: Image.Image, semaphore: asyncio.Semaphore) -> list[Block]:
        layout_img = image.resize(self.layout_image_size, Image.Resampling.BICUBIC)
        layout_out = await self._chat(layout_img, self.layout_prompt, "layout", semaphore)
        blocks = parse_layout_output(layout_out)

        async def recognize(b: ContentBlock) -> None:
            crop = crop_block(image, b)
            if crop is None:
                return
            prompt = self.prompts.get(b.type) or self.prompts["default"]
            b.content = await self._chat(crop, prompt, b.type, semaphore)

        await asyncio.gather(*(recognize(b) for b in blocks if b.type not in _SKIP_EXTRACT))

        try:
            blocks = post_process(
                blocks, simple_post_process=False, handle_equation_block=True,
                abandon_list=self.abandon_list, abandon_paratext=self.abandon_paratext,
            )
        except Exception as e:  # noqa: BLE001 - match upstream: fall back to raw blocks
            logger.warning("TeleOCR post-processing failed: {}", e)
        return [Block(type=b.type, bbox=_outer_rect(b.bbox), content=b.content, angle=b.angle) for b in blocks]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _outer_rect(coords: list[float]) -> list[float]:
    xs, ys = coords[0::2], coords[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]
