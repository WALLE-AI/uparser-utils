"""Crop non-text blocks (figures, seals, scientific charts, table screenshots) so that markdown /
middle.json can reference them, mirroring MinerU's cut_image_and_table()."""
from __future__ import annotations

import hashlib
import io

from PIL import Image

from .models import Block

_CROP_TYPES = {"image", "seal", "char"}


def extract_assets(img: Image.Image, blocks: list[Block], quality: int = 90) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    W, H = img.size
    for b in blocks:
        if b.type not in _CROP_TYPES and not (b.type == "table" and not (b.content or "").strip()):
            continue
        xs, ys = b.bbox[0::2], b.bbox[1::2]
        x1, y1, x2, y2 = (int(min(xs) * W), int(min(ys) * H), int(max(xs) * W), int(max(ys) * H))
        if x2 <= x1 or y2 <= y1:
            continue
        buf = io.BytesIO()
        img.crop((x1, y1, x2, y2)).save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        rel = f"{b.type}/{hashlib.sha256(data).hexdigest()[:32]}.jpg"
        assets[rel] = data
        b.image_path = rel
    return assets
