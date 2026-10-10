"""Input loading: PDF / image bytes -> PIL pages."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

from PIL import Image

_PDF_MAGIC = b"%PDF-"
_PDF_HEADER_WINDOW = 1024   # spec: header may follow up to 1024 bytes of junk; pdfium accepts that


PageIn = tuple[int, Image.Image, tuple[int, int]]   # (index, RGB image, page size in PDF points / px for images)


def load_pages(source: str | Path | bytes | Image.Image, *, dpi: int = 150,
               pages: Sequence[int] | None = None) -> list[PageIn]:
    """Return [(page_index, RGB image, (w, h) in PDF points)]. `pages` is a 0-based whitelist (PDF only)."""
    if isinstance(source, Image.Image):
        img = source.convert("RGB")
        return [(0, img, img.size)]
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    if _PDF_MAGIC in data[:_PDF_HEADER_WINDOW]:
        return _render_pdf(data, dpi, pages)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return [(0, img, img.size)]


def _render_pdf(data: bytes, dpi: int, pages: Sequence[int] | None) -> list[PageIn]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        wanted = range(len(pdf)) if pages is None else [p for p in pages if 0 <= p < len(pdf)]
        out: list[PageIn] = []
        for i in wanted:
            page = pdf[i]
            w, h = page.get_size()
            out.append((i, page.render(scale=dpi / 72).to_pil().convert("RGB"), (int(w), int(h))))
        return out
    finally:
        pdf.close()
