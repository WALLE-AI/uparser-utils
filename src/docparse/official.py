"""MinerU 4.0-aligned output (model_list -> ModelJson -> MiddleJson -> markdown / structured content).

Mirrors MinerU 4.0.11 `backend/analysis/pdf` + `backend/postprocess` for the `xhigh` (pure VLM) tier:

    VLM blocks --_convert_vlm_results_to_model_list--> --_normalize_xhigh_vlm_blocks--> title typing
      --attach_visual_block_images (image_base64 crops)--> _normalize_pdf_model_list --> ModelJson
      --docvortex.model_json_to_middle_json--> MiddleJson --docvortex.render--> md / structured_content

Everything after the VLM call is the same code as upstream (docvortex + vendored normalization).
Stages that need MinerU's *local* GPU models are NOT reproduced: the small layout model (doc_title vs
paragraph_title split), OCR det (line boxes -> block bbox is used as a single line), formula recognition
and native PDF text-layer filling.
"""
from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from PIL import Image

from .models import Block, Document

_warned = False


def available() -> bool:
    try:
        import docvortex  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_style(style: str) -> str:
    global _warned
    if style not in ("auto", "v4", "legacy"):
        raise ValueError("style must be 'auto', 'v4' or 'legacy'")
    if style == "auto":
        if available():
            return "v4"
        if not _warned:
            logger.warning("docvortex not installed: using legacy MinerU output. "
                           "pip install 'docparse-sdk[official]' for MinerU 4.0-aligned output.")
            _warned = True
        return "legacy"
    if style == "v4" and not available():
        raise ImportError("style='v4' needs: pip install 'docparse-sdk[official]'")
    return style


# ---- VLM block -> raw model_list entry ----------------------------------------------------
_TELEOCR_TYPE_MAP = {"char": "chart"}          # scientific figure
_CODE_PREFIX = re.compile(r"^<_([^>]*?)_>")
# MinerU's model emits inline math as \( ... \); TeleOCR emits $...$ (paired by its own post-process).
_INLINE_DOLLAR = re.compile(r"(?<![\\$])\$(?!\$)(.+?)(?<![\\$])\$(?!\$)", re.S)
_EQ_TAG = re.compile(r"^\s*\$+(.*?)\$+\s*[(（]\s*([^()（）\s]+)\s*[)）]\s*$", re.S)
_NATURAL_TYPES = {"text", "title", "list", "ref_text", "header", "footer", "page_number", "aside_text",
                  "page_footnote", "table_caption", "image_caption", "code_caption", "table_footnote",
                  "image_footnote", "index", "phonetic"}


def _teleocr_inline_math(text: str) -> str:
    if len(re.findall(r"(?<![\\$])\$(?!\$)", text)) % 2:      # unbalanced dollars: not math, leave as text
        return text
    return _INLINE_DOLLAR.sub(lambda m: "\\(" + m.group(1).strip() + "\\)", text)


def _teleocr_equation(content: str) -> str:
    """TeleOCR post-process yields `$$x$$`, `$x$` or `$x\\quad$ (1)`; upstream wants bare LaTeX, with the
    equation number carried as `\\tag {n}` (as MinerU's own model does)."""
    content = content.strip()
    m = _EQ_TAG.match(content)
    if m:
        core, num = m.group(1).strip(), m.group(2)
        core = re.sub(r"(\\q?quad\s*)+$", "", core).strip()
        return f"{core} \\tag {{{num}}}"
    while content.startswith("$") and content.endswith("$") and len(content) >= 2:
        content = content[1:-1].strip()
    return re.sub(r"(\\q?quad\s*)+$", "", content).strip()


def block_to_raw(b: Block, backend: str) -> dict[str, Any] | None:
    """Adapt a backend Block to the dict shape mineru-vl-utils' ContentBlock produces."""
    t, content, extra = b.type, b.content, dict(b.extra)
    if t in _NATURAL_TYPES and t != "list" and not (content or "").strip():
        return None                # nothing to render; upstream validation rejects non-span (None) inline content
    if t == "equation" and not (content or "").strip():
        return None
    if backend == "teleocr":
        if t == "equation_block":
            return None
        if t == "list" and not (content or "").strip():
            return None            # TeleOCR does not extract list regions; its child text blocks remain
        if t == "seal":
            t, extra["sub_type"] = "image", "seal"
        elif t in _TELEOCR_TYPE_MAP:
            t = _TELEOCR_TYPE_MAP[t]
        elif t == "algorithm":
            t = "code"
        if t == "equation" and content:
            content = _teleocr_equation(content)
        elif t in _NATURAL_TYPES and content:
            content = _teleocr_inline_math(content)
        if t == "code" and content:
            content = _CODE_PREFIX.sub("", content, count=1)
    raw: dict[str, Any] = {"type": t, "bbox": list(b.bbox), "content": content, "angle": b.angle or 0}
    raw.update(extra)
    return raw


TitleClassifier = Callable[[dict[str, Any], int], tuple[str, int]]


def default_title_classifier(block: dict[str, Any], page_index: int) -> tuple[str, int]:
    """Upstream splits VLM `title` into doc_title(level 1)/paragraph_title(level 2) by overlap with
    its local layout model's doc_title boxes. Without that model every title is a paragraph_title."""
    return "paragraph_title", 2


def build_raw_page(image: Image.Image, blocks: list[Block], backend: str, page_index: int,
                   title_classifier: TitleClassifier = default_title_classifier) -> list[dict[str, Any]]:
    from docvortex.document.pdf.visuals import attach_visual_block_images

    from ._vendor.mineru4.layout import _convert_vlm_results_to_model_list, _normalize_xhigh_vlm_blocks

    raws = [r for r in (block_to_raw(b, backend) for b in blocks) if r is not None]
    page = _convert_vlm_results_to_model_list([raws])
    _normalize_xhigh_vlm_blocks(page)
    for blk in page[0]:
        if blk.get("type") == "title":
            blk["type"], blk["level"] = title_classifier(blk, page_index)
    attach_visual_block_images(page, [{"img_pil": image}], page_start_index=page_index)
    return page[0]


# ---- document level -------------------------------------------------------------------------
def build_model_json(doc: Document):
    """Cached per Document (normalization is destructive on its own copy and logs warnings)."""
    if "model_json" not in doc._cache:
        doc._cache["model_json"] = _build_model_json(doc)
    return doc._cache["model_json"]


def _build_model_json(doc: Document):
    from docvortex.schema import DocumentMetadata, ModelJson, Producer

    from . import __version__
    from ._vendor.mineru4.normalization import _normalize_pdf_model_list

    if any(p.raw is None for p in doc.pages):
        raise ValueError("document was parsed without MinerU 4.0 raw blocks (docvortex missing at parse time?)")
    pages = [deepcopy(p.raw) for p in doc.pages]
    _normalize_pdf_model_list(pages)
    return ModelJson(
        pages=pages,
        page_index_map=[p.index for p in doc.pages] if doc.partial else [],
        metadata=DocumentMetadata(
            file_suffix="pdf" if doc.source.lower().endswith(".pdf") or not doc.source else "pdf",
            producer=Producer(name="docparse-sdk", version=__version__),
        ),
        extensions={"docparse": {"backend": doc.backend, "aligned_with": "mineru-4.0.11/xhigh"}},
    )


def build_middle_json(doc: Document):
    if "middle_json" not in doc._cache:
        from docvortex.postprocess.document import model_json_to_middle_json

        doc._cache["middle_json"] = model_json_to_middle_json(build_model_json(doc))
    return doc._cache["middle_json"]


def to_middle_json_dict(doc: Document) -> dict[str, Any]:
    return build_middle_json(doc).to_dict(exclude_block_fields={"image_base64"})


def to_markdown(doc: Document, *, asset_base_url: str = "", full: bool = False, **_: Any) -> str:
    from docvortex.render import render_markdown
    from docvortex.render.contracts import RenderMode

    return render_markdown(build_middle_json(doc), mode=RenderMode.FULL if full else RenderMode.DEFAULT,
                           asset_base_url=asset_base_url)


def to_structured_content(doc: Document, *, asset_base_url: str = "") -> dict[str, Any]:
    from docvortex.render import render_structured_content

    return render_structured_content(build_middle_json(doc), asset_base_url=asset_base_url)


def save_bundle(doc: Document, out_dir: Path, *, with_assets: bool = True) -> dict[str, Path]:
    """Same files as MinerU 4.0 `ParseResult.save()`: image sidecars are externalized from image_base64."""
    import json

    from docvortex.export import materialize_middle, validate_materialized_assets
    from docvortex.render import render_markdown, render_structured_content

    model_json = build_model_json(doc)
    middle, assets = materialize_middle(build_middle_json(doc))
    validate_materialized_assets(middle, assets)
    texts = {
        "markdown.md": render_markdown(middle),
        "middle_json.json": json.dumps(middle.to_dict(exclude_block_fields={"image_base64"}),
                                       ensure_ascii=False, indent=2),
        "structured_content.json": json.dumps(render_structured_content(middle), ensure_ascii=False, indent=2),
        "model_output.json": json.dumps(model_json.to_dict(skip_defaults=False), ensure_ascii=False, indent=2),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, text in texts.items():
        (out_dir / name).write_text(text, encoding="utf-8")
        paths[name] = out_dir / name
    if with_assets:
        for rel, data in sorted(assets.items()):
            f = out_dir / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
            paths[rel] = f
    return paths
