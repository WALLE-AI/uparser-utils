import io
import json
import re
from pathlib import Path

import pytest
from PIL import Image

from docparse import AsyncDocParser, DocParser
from .conftest import SPEC, URLS

pytest.importorskip("docvortex")
from docvortex.schema import MiddleJson  # noqa: E402

pytestmark = pytest.mark.e2e


def types(doc):
    return {b.type for p in doc.pages for b in p.blocks}


def test_block_coverage(paper_doc):
    t = types(paper_doc)
    assert {"text", "title", "image", "table", "equation"} <= t, t
    assert [p.index for p in paper_doc.pages] == [0, 2, 3, 7, 9] and paper_doc.partial
    for p in paper_doc.pages:
        assert p.blocks and p.page_size and p.raw is not None
        for b in p.blocks:
            assert 0 <= min(b.bbox) and max(b.bbox) <= 1 and b.bbox[0] < b.bbox[2] and b.bbox[1] < b.bbox[3]


def test_equation_and_table_content(paper_doc):
    eq = [b.content for p in paper_doc.pages for b in p.blocks if b.type == "equation"]
    assert eq and all(e and len(e.strip()) > 3 for e in eq)
    tables = [b.content for p in paper_doc.pages for b in p.blocks if b.type == "table"]
    assert tables and any("<table" in t and "<td" in t for t in tables)
    md = paper_doc.to_markdown("v4")
    assert "$$" in md or "\\[" in md or "$" in md          # LaTeX delimiters rendered
    assert "<table" in md


def test_math_delimiters_clean(paper_doc):
    """No escaped dollars, and display equations must not contain nested $ delimiters."""
    md = re.sub(r"!\[\]\(data:image[^)]*\)", "", paper_doc.to_markdown("v4"))
    assert "\\$" not in md, re.findall(r".{20}\\\$.{20}", md)[:3]
    displays = re.findall(r"\$\$\n(.*?)\n\$\$", md, re.S)
    assert displays and all("$" not in d for d in displays), displays[:2]
    assert any("\\tag" in d for d in displays)                 # equation numbers kept as \tag{n}


def test_v4_bundle_files_and_images(paper_doc, tmp_path):
    paths = paper_doc.save(tmp_path, style="v4")
    for f in ("markdown.md", "middle_json.json", "structured_content.json", "model_output.json"):
        assert (tmp_path / f).stat().st_size > 100, f
    imgs = [k for k in paths if k.startswith("images/")]
    assert len(imgs) >= 3
    for k in imgs:                                  # every exported crop is a real, non-trivial image
        im = Image.open(tmp_path / k)
        im.load()
        assert im.width > 10 and im.height > 10
    md = (tmp_path / "markdown.md").read_text(encoding="utf-8")
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)
    assert refs and all((tmp_path / r).exists() for r in refs), refs
    mid = json.loads((tmp_path / "middle_json.json").read_text(encoding="utf-8"))
    assert "image_base64" not in json.dumps(mid) and "data:image" not in json.dumps(mid)
    assert len(MiddleJson.from_dict(mid).pages) == 5              # strict upstream schema round-trip


def test_model_json_normalized(paper_doc):
    mo = paper_doc.to_model_json()
    assert mo["page_index_map"] == [0, 2, 3, 7, 9]
    for page in mo["pages"]:
        for blk in page:
            assert "angle" not in blk and "merge_prev" not in blk and blk["type"] != "title"
            if isinstance(blk.get("content"), list):                # inline content is a span list
                assert all("type" in s for s in blk["content"])


def test_structured_content(paper_doc):
    sc = paper_doc.to_structured_content()
    assert sc and len(json.dumps(sc)) > 500


def test_legacy_style(paper_doc, tmp_path):
    mj = paper_doc.to_middle_json("legacy")
    assert len(mj["pdf_info"]) == 5
    kinds = {b["type"] for pi in mj["pdf_info"] for b in pi["para_blocks"]}
    assert {"text", "title", "image", "table", "interline_equation"} <= kinds, kinds
    img_blocks = [b for pi in mj["pdf_info"] for b in pi["para_blocks"] if b["type"] == "image"]
    assert img_blocks and all(any(x["type"] == "image_body" for x in b["blocks"]) for b in img_blocks)
    paths = paper_doc.save(tmp_path, style="legacy")
    md = paths["md"].read_text(encoding="utf-8")
    refs = re.findall(r"!\[\]\(([^)]+)\)", md)
    assert refs and all((tmp_path / r).exists() for r in refs), refs


def test_caption_association(paper_doc):
    """Figures/tables keep their captions inside the same parent block in the v4 middle json."""
    mid = paper_doc.to_middle_json("v4")
    parents = [b for p in mid["pages"] for b in p["blocks"] if b["type"] in ("image", "table")]
    assert parents
    assert any(isinstance(b.get("content"), list) and any(c["type"].endswith("caption") for c in b["content"])
               for b in parents)


def test_png_input(backend):
    import pypdfium2 as pdfium

    img = pdfium.PdfDocument(SPEC)[61].render(scale=2).to_pil()
    buf = io.BytesIO()
    img.save(buf, "PNG")
    doc = DocParser(backend, base_url=URLS[backend]).parse(buf.getvalue())
    assert len(doc.pages) == 1 and not doc.partial and "<table" in doc.to_markdown("v4")


async def test_async_batch(backend):
    async with AsyncDocParser(backend, base_url=URLS[backend], concurrency=32) as p:
        docs = await p.parse_many([SPEC, SPEC], pages=[20, 21])
    assert [len(d.pages) for d in docs] == [2, 2]
    assert all(len(d.to_markdown("v4")) > 300 for d in docs)


def test_backends_agree_on_text():
    """Where both services are up, they should extract near-identical text modulo punctuation width."""
    import difflib

    from .conftest import _up
    if not all(_up(u) for u in URLS.values()):
        pytest.skip("need both services")
    g = lambda b: "".join(x.content or "" for x in DocParser(b, base_url=URLS[b]).parse(SPEC, pages=[21]).pages[0].blocks
                          if x.type == "text")
    norm = lambda s: "".join(c for c in s if c.isalnum())
    assert difflib.SequenceMatcher(None, norm(g("mineru")), norm(g("teleocr")), autojunk=False).ratio() > 0.95


def test_code_block_page(backend):
    """Synthetic page with a Python snippet: code type, fence language, indentation survive both styles."""
    import glob

    from PIL import ImageDraw, ImageFont
    fonts = glob.glob("/usr/share/fonts/**/DejaVuSansMono.ttf", recursive=True) + glob.glob("/usr/share/fonts/**/*Mono*.ttf", recursive=True)
    if not fonts:
        pytest.skip("no monospace font available")
    f, big = ImageFont.truetype(fonts[0], 26), ImageFont.truetype(fonts[0], 40)
    img = Image.new("RGB", (1240, 1000), "white")
    d = ImageDraw.Draw(img)
    d.text((80, 60), "3.2 Example implementation", fill="black", font=big)
    code = ["def fib(n: int) -> int:", "    a, b = 0, 1", "    for _ in range(n):", "        a, b = b, a + b", "    return a"]
    d.rectangle((70, 140, 1170, 400), outline="gray", width=2)
    for i, line in enumerate(code):
        d.text((90, 150 + i * 38), line, fill="black", font=f)
    d.text((80, 460), "The function above runs in linear time.", fill="black", font=f)
    doc = DocParser(backend, base_url=URLS[backend]).parse(img)
    assert "code" in types(doc)
    for style in ("v4", "legacy"):
        md = doc.to_markdown(style)
        assert "```" in md and "def fib(n: int) -> int:" in md and "        a, b = b, a + b" in md, (style, md)
    assert "```python" in doc.to_markdown("v4")
