"""Document -> Markdown. Deliberately simple: blocks are emitted in layout order."""
from __future__ import annotations

from .models import Block, Document

_DROP_DEFAULT = {"header", "footer", "page_number", "aside_text", "unknown"}


def _fmt(b: Block) -> str:
    c = (b.content or "").strip()
    t = b.type
    if t == "title":
        return f"# {c}" if c else ""
    if t == "equation":
        return c if c.startswith("$$") else f"$$\n{c}\n$$" if c else ""
    if t == "code" or t == "algorithm":
        return f"```\n{c}\n```" if c else ""
    if t == "image":
        return "![image]()"
    if t in ("image_caption", "table_caption", "code_caption"):
        return f"*{c}*" if c else ""
    if t == "list":
        return c
    return c  # text, table (HTML), footnotes, ref_text, ...


def to_markdown(doc: Document, *, drop_types: set[str] | None = None, page_separator: str = "\n\n") -> str:
    drop = _DROP_DEFAULT if drop_types is None else drop_types
    pages = []
    for p in doc.pages:
        parts = [s for b in p.blocks if b.type not in drop and (s := _fmt(b))]
        pages.append("\n\n".join(parts))
    return page_separator.join(s for s in pages if s)
