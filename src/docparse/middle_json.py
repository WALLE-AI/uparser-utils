"""Blocks -> MinerU-style middle.json -> Markdown.

Follows MinerU's VLM post-processing chain (vendored from TeleOCR, itself a fork of MinerU):
  model blocks --MagicModel--> para_blocks (list / image+caption / table+caption+footnote /
  code / title ... grouped by spatial association) --> middle.json --union_make--> Markdown.
"""
from __future__ import annotations

from typing import Any

from .models import Block, Document, Page
from ._vendor.teleocr.enum_class import ContentType
from ._vendor.teleocr.magic_model import MagicModel
from ._vendor.teleocr.mkcontent import union_make
from ._vendor.teleocr._lite import detect_lang

MIDDLE_JSON_VERSION = "docparse-vlm"


def _merge_prev(blocks: list[Block]) -> list[Block]:
    """MinerU layout marks a text block as continuing the previous one (paragraph split by a column
    or figure). Fold such blocks into the previous text block (same page only)."""
    out: list[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
        if b.type == "text" and b.merge_prev and prev is not None and prev.type == "text":
            a, c = (prev.content or "").rstrip(), (b.content or "").lstrip()
            sep = "" if detect_lang(a + c) in ("zh", "ja", "ko") else " "
            merged = Block(**{**prev.__dict__, "content": a + sep + c})
            merged.bbox = [min(prev.bbox[0], b.bbox[0]), min(prev.bbox[1], b.bbox[1]),
                           max(prev.bbox[2], b.bbox[2]), max(prev.bbox[3], b.bbox[3])]
            out[-1] = merged
        else:
            out.append(b)
    return out


def _block_dict(b: Block) -> dict[str, Any]:
    return {"type": b.type, "bbox": b.bbox, "angle": b.angle, "content": b.content, "image_path": b.image_path}


def page_to_info(page: Page) -> dict[str, Any]:
    width, height = page.page_size or (page.width, page.height)
    magic = MagicModel([_block_dict(b) for b in _merge_prev(page.blocks)], width, height)
    for span in magic.get_all_spans():
        if span["type"] in (ContentType.IMAGE, ContentType.SEAL, ContentType.CHAR) and len(span["bbox"]) > 4:
            xs, ys = span["bbox"][0::2], span["bbox"][1::2]
            span["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
    return {
        "para_blocks": magic.get_page_blocks(),
        "discarded_blocks": [],
        "page_size": [width, height],
        "page_idx": page.index,
    }


def to_middle_json(doc: Document) -> dict[str, Any]:
    from . import __version__

    return {
        "pdf_info": [page_to_info(p) for p in doc.pages],
        "_backend": "vlm",
        "_model": doc.backend,
        "_version_name": f"docparse-sdk {__version__}",
    }


def to_markdown(doc: Document, img_dir: str = "images") -> str:
    return union_make(to_middle_json(doc)["pdf_info"], img_dir)
