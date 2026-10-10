"""Offline stand-ins for the vLLM OpenAI-compatible services."""
import asyncio
import json

import httpx
import respx

TELEOCR_LAYOUT_PROMPT = "Analyze the image layout."
MINERU_LAYOUT_PROMPT = "Layout Detection:"


def prompt_of(request: httpx.Request) -> str:
    return json.loads(request.content)["messages"][-1]["content"][-1]["text"]


def payload_of(request: httpx.Request) -> dict:
    return json.loads(request.content)


def chat(text: str, status: int = 200) -> httpx.Response:
    if status != 200:
        return httpx.Response(status, json={"error": "boom"})
    return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def mount(base: str, handler, models=("m",)) -> respx.Route:
    respx.get(f"{base}/v1/models").mock(return_value=httpx.Response(
        200, json={"object": "list", "data": [{"id": m} for m in models]}))
    return respx.post(f"{base}/v1/chat/completions").mock(side_effect=handler)


class Tracker:
    """Records requests and the peak number of in-flight ones (for concurrency-limit tests)."""

    def __init__(self, handler, delay=0.0):
        self.handler, self.delay, self.inflight, self.peak, self.calls = handler, delay, 0, 0, []

    async def __call__(self, request):
        self.calls.append(request)
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return self.handler(request)
        finally:
            self.inflight -= 1


def mineru_layout(*items) -> str:
    """items: (x1,y1,x2,y2,type[,tail])"""
    out = []
    for it in items:
        x1, y1, x2, y2, t, *tail = it
        out.append(f"<|box_start|>{x1} {y1} {x2} {y2}<|box_end|><|ref_start|>{t}<|ref_end|><|rotate_up|>{tail[0] if tail else ''}")
    return "".join(out)


def teleocr_layout(*items) -> str:
    """items: (coords_str, label[, angle])"""
    return "\n".join(f"<box:{c}><label:{l}><{(a[0] if a else 'up')}>" for c, l, *a in items)
