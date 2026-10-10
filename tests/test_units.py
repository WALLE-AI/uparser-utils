import asyncio
import io
import json

import httpx
import pytest
import respx
from PIL import Image

from docparse import AsyncDocParser, BackendConfig, Block, DocParser, Document, Page
from docparse import official
from docparse.assets import extract_assets
from docparse.backends import create_backend
from docparse.backends.teleocr import crop_block, parse_layout_output, _resize_by_need, TeleOCRBackend
from docparse.middle_json import _merge_prev
from docparse.pdf import load_pages
from docparse.render import to_markdown as simple_md

from .fakes import (MINERU_LAYOUT_PROMPT, TELEOCR_LAYOUT_PROMPT, Tracker, chat, mineru_layout, mount, payload_of,
                    prompt_of, teleocr_layout)

pytest.importorskip("docvortex")
BASE = "http://t"


def page_img(w=400, h=600, color="white"):
    return Image.new("RGB", (w, h), color)


# ---------------------------------------------------------------- config / factory
def test_config_url_and_headers():
    assert BackendConfig(base_url="http://h:1/v1/").normalized_url() == "http://h:1"
    assert BackendConfig(base_url="http://h:1").normalized_url() == "http://h:1"
    h = BackendConfig(base_url="x", api_key="k", headers={"X-A": "1"}).request_headers()
    assert h == {"X-A": "1", "Authorization": "Bearer k"}
    assert BackendConfig(base_url="x", api_key="k", headers={"Authorization": "Own"}).request_headers()["Authorization"] == "Own"


def test_factory():
    assert create_backend("TeleOCR").config.base_url.endswith(":8004")
    assert create_backend("mineru").config.base_url.endswith(":8003")
    with pytest.raises(ValueError, match="unknown backend"):
        create_backend("nope")
    with pytest.raises(ValueError, match="unknown backend"):
        create_backend("nope", BackendConfig(base_url="x"))


# ---------------------------------------------------------------- input loading
def test_load_pages_image_png_and_pdf(tmp_path):
    im = page_img(300, 200)
    assert load_pages(im) == [(0, im.convert("RGB"), (300, 200))] or load_pages(im)[0][2] == (300, 200)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    assert load_pages(buf.getvalue())[0][1].size == (300, 200)
    pdf = tmp_path / "a.pdf"
    page_img(595, 842).save(pdf, "PDF", save_all=True, append_images=[page_img(595, 842), page_img(595, 842)])
    pages = load_pages(pdf, dpi=72)
    assert [p[0] for p in pages] == [0, 1, 2] and pages[0][1].size == pages[0][2]
    big = load_pages(pdf, dpi=144, pages=[2, 99, -1])          # out-of-range / negative filtered
    assert [p[0] for p in big] == [2] and big[0][1].width == 2 * big[0][2][0]
    assert load_pages(pdf.read_bytes(), pages=[1])[0][0] == 1  # bytes source


# ---------------------------------------------------------------- teleocr layout parsing + crops
def test_parse_layout_edge_cases():
    out = "\n".join([
        "<box:0 0 1000 1000><label:text><up>",             # full page, valid
        "<box:100 100 200 100 200 200 100 200><label:table><right>",  # polygon, 4 points
        "<box:100 100 1001 200><label:text><up>",          # >1000 -> invalid
        "<box:100 100 200><label:text><up>",               # odd count
        "<box:100 100 200 200><label:nonsense><up>",       # unknown label
        "<box:100 100 200 200><label:image_caption><down>",
        "", "   ", "free text line",
        "<box:1 2 3 4><label:TITLE><left>",                # label case-insensitive
        "<box:1 2 3 4><label:header><weird>",              # no angle token
    ])
    bl = parse_layout_output(out)
    assert [(b.type, b.angle, len(b.bbox)) for b in bl] == [
        ("text", 0, 4), ("table", 90, 8), ("image_caption", 180, 4), ("title", 270, 4), ("header", None, 4)]
    assert bl[0].bbox == [0.0, 0.0, 1.0, 1.0]


def test_crop_block_variants():
    img = Image.new("RGB", (1000, 1000), "red")
    from docparse._vendor.teleocr.structs import ContentBlock
    rect = ContentBlock("text", [0.1, 0.1, 0.5, 0.3], angle=0)
    c = crop_block(img, rect)
    assert c.size == (400, 200)
    rot = ContentBlock("text", [0.1, 0.1, 0.5, 0.3], angle=90)
    assert crop_block(img, rot).size == (200, 400)
    poly = ContentBlock("text", [0.0, 0.0, 0.4, 0.0, 0.0, 0.4], angle=0)    # triangle
    pc = crop_block(img, poly)
    assert pc.getpixel((1, 1)) == (255, 0, 0) and pc.getpixel((pc.width - 2, pc.height - 2)) == (0, 0, 0)
    assert crop_block(img, ContentBlock("text", [0.5, 0.5, 0.5001, 0.9], angle=0)) is None   # <1px wide


def test_resize_by_need():
    thin = Image.new("RGB", (1000, 4), "white")
    r = _resize_by_need(thin)
    assert max(r.size) / min(r.size) <= 50 and min(r.size) >= 28        # padded (ratio) then upscaled (min edge)
    assert Image.new("RGB", (100, 50), "white").size == _resize_by_need(Image.new("RGB", (100, 50), "white")).size
    tiny = Image.new("RGB", (10, 10), "white")
    assert min(_resize_by_need(tiny).size) >= 28


# ---------------------------------------------------------------- teleocr backend over HTTP
def tele_cfg(**kw):
    return BackendConfig(base_url=BASE, model="teleocr", max_retries=0, **kw)


@respx.mock
async def test_teleocr_request_payload_and_prompts():
    seen = []

    def h(req):
        seen.append(payload_of(req))
        p = prompt_of(req)
        if TELEOCR_LAYOUT_PROMPT in p:
            return chat(teleocr_layout(("100 100 900 200", "title"), ("100 300 900 400", "table"),
                                       ("100 500 900 600", "equation"), ("100 700 900 800", "code"),
                                       ("100 820 900 900", "text")))
        return chat("x")

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=h)
    async with AsyncDocParser("teleocr", config=tele_cfg(prompts={"text": "\nCUSTOM TEXT"}), no_repeat_ngram_size=100) as p:
        await p.parse(page_img())
    by_prompt = {s["messages"][-1]["content"][-1]["text"]: s for s in seen}
    assert "\nThis is the image of a table. Please output the table in OTSL format." in by_prompt
    assert "\nPlease write out the expression of the formula in the image using LaTeX format." in by_prompt
    assert "\nThe image contains a code snippet, please output the parsing result." in by_prompt
    assert "\nCUSTOM TEXT" in by_prompt                    # override applied to text/title/default... via 'text'
    s = by_prompt["\nThis is the image of a table. Please output the table in OTSL format."]
    assert s["skip_special_tokens"] is False and s["top_k"] == 1 and s["temperature"] == 0.0
    assert s["presence_penalty"] == 1.0 and s["frequency_penalty"] == 0.005          # table sampling profile
    assert s["vllm_xargs"] == {"no_repeat_ngram_size": 100}
    assert s["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    lay = by_prompt["\nAnalyze the image layout."]
    assert lay["presence_penalty"] == 0.0 and "vllm_xargs" in lay


@respx.mock
async def test_teleocr_no_xargs_by_default_and_model_autodetect():
    seen = []
    mount(BASE, lambda r: (seen.append(payload_of(r)), chat(teleocr_layout(("0 0 500 500", "text"))
                                                       if TELEOCR_LAYOUT_PROMPT in prompt_of(r) else "hi"))[1], models=("auto-model",))
    async with AsyncDocParser("teleocr", config=BackendConfig(base_url=BASE, max_retries=0)) as p:
        d = await p.parse(page_img())
    assert all("vllm_xargs" not in s for s in seen) and {s["model"] for s in seen} == {"auto-model"}
    assert d.pages[0].blocks[0].content == "hi"


@respx.mock
async def test_teleocr_server_errors_propagate():
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=lambda r: chat("", status=500))
    async with AsyncDocParser("teleocr", config=tele_cfg()) as p:
        with pytest.raises(httpx.HTTPStatusError):
            await p.parse(page_img())


@respx.mock
async def test_teleocr_garbage_layout_gives_empty_page_and_skips():
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=lambda r: chat("I cannot do that"))
    async with AsyncDocParser("teleocr", config=tele_cfg()) as p:
        d = await p.parse(page_img())
    assert d.pages[0].blocks == [] and d.to_markdown("legacy") == "" and d.to_markdown("v4") == ""


@respx.mock
async def test_teleocr_skips_image_list_and_postprocess_flags():
    layout = teleocr_layout(("100 100 900 200", "image"), ("100 220 900 300", "list"), ("100 320 900 400", "header"),
                            ("100 420 900 500", "text"))
    tr = Tracker(lambda r: chat(layout if TELEOCR_LAYOUT_PROMPT in prompt_of(r) else "T"))
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=tr)
    async with AsyncDocParser("teleocr", config=tele_cfg(), abandon_paratext=True, abandon_list=True) as p:
        d = await p.parse(page_img())
    assert len(tr.calls) == 1 + 2                       # layout + header + text only (image/list not extracted)
    assert [b.type for b in d.pages[0].blocks] == ["image", "text"]     # header + list abandoned


@respx.mock
async def test_concurrency_limit_is_respected():
    layout = teleocr_layout(*[(f"100 {i*50+10} 900 {i*50+40}", "text") for i in range(12)])
    tr = Tracker(lambda r: chat(layout if TELEOCR_LAYOUT_PROMPT in prompt_of(r) else "T"), delay=0.02)
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=tr)
    async with AsyncDocParser("teleocr", config=tele_cfg(concurrency=3)) as p:
        await p.parse_many([page_img(), page_img()])
    assert 1 < tr.peak <= 3, tr.peak


@respx.mock
async def test_page_order_preserved_under_concurrency(tmp_path):
    import random
    pdf = tmp_path / "a.pdf"
    page_img(300, 300).save(pdf, "PDF", save_all=True, append_images=[page_img(300, 300) for _ in range(5)])

    async def h(req):
        await asyncio.sleep(random.random() * 0.03)
        return chat(teleocr_layout(("100 100 900 900", "text")) if TELEOCR_LAYOUT_PROMPT in prompt_of(req) else "t")

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=h)
    async with AsyncDocParser("teleocr", config=tele_cfg(concurrency=8), dpi=72) as p:
        d = await p.parse(pdf)
    assert [pg.index for pg in d.pages] == list(range(6)) and all(b.page == pg.index for pg in d.pages for b in pg.blocks)


# ---------------------------------------------------------------- mineru backend over HTTP (real mineru-vl-utils client)
@respx.mock
async def test_mineru_backend_full_flow():
    layout = mineru_layout((100, 100, 900, 200, "title"), (100, 220, 900, 400, "text"),
                           (100, 420, 900, 500, "text", "<|txt_contd_tgt|>"), (100, 520, 900, 700, "table"),
                           (100, 720, 900, 800, "equation"))

    def h(req):
        p = prompt_of(req)
        if MINERU_LAYOUT_PROMPT in p:
            return chat(layout)
        if "Table Recognition" in p:
            return chat("<fcel>a<fcel>b<nl><fcel>1<fcel>2<nl>")
        if "Formula Recognition" in p:
            return chat("x^{2}")
        return chat("para")

    mount(BASE, h, models=("mineru2.5",))
    async with AsyncDocParser("mineru", config=BackendConfig(base_url=BASE, model="mineru2.5", max_retries=0)) as p:
        d = await p.parse(page_img())
    blocks = d.pages[0].blocks
    assert [b.type for b in blocks] == ["title", "text", "text", "table", "equation"]
    assert blocks[2].merge_prev and not blocks[1].merge_prev
    assert "<table" in blocks[3].content and "x^{2}" in blocks[4].content
    assert all("rotate" not in k for b in blocks for k in b.extra)
    md = d.to_markdown("legacy")
    assert md.count("para") >= 1 and "<table" in md
    assert "## " in d.to_markdown("v4")
    eq = [b for b in d.to_model_json()["pages"][0] if b["type"] == "equation"]
    assert eq[0]["content"] == "x^{2}"                      # upstream normalization strips mineru-vl-utils' \\[ \\] wrapper


# ---------------------------------------------------------------- assets / merge / render
def test_extract_assets_rules():
    img = Image.new("RGB", (200, 200), "blue")
    bl = [Block("image", [0.1, 0.1, 0.5, 0.5]), Block("seal", [0.0, 0.0, 0.2, 0.0, 0.2, 0.2, 0.0, 0.2]),
          Block("table", [0.5, 0.5, 0.9, 0.9], content=""), Block("table", [0.5, 0.5, 0.9, 0.9], content="<table/>"),
          Block("text", [0.0, 0.0, 1.0, 1.0], content="t"), Block("char", [0.3, 0.3, 0.3, 0.9])]       # degenerate
    assets = extract_assets(img, bl)
    assert bl[0].image_path and bl[1].image_path and bl[2].image_path          # empty table gets a screenshot
    assert bl[3].image_path is None and bl[4].image_path is None and bl[5].image_path is None
    assert all(k.endswith(".jpg") and v[:2] == b"\xff\xd8" for k, v in assets.items())
    assert bl[0].image_path.startswith("image/") and bl[1].image_path.startswith("seal/")


def test_merge_prev_rules():
    t = lambda c, mp=False, ty="text": Block(ty, [0.1, 0.1, 0.5, 0.2], content=c, merge_prev=mp)
    zh = _merge_prev([t("你好"), t("世界", True)])
    assert len(zh) == 1 and zh[0].content == "你好世界"
    en = _merge_prev([t("hello"), t("world", True)])
    assert en[0].content == "hello world"
    assert len(_merge_prev([t("a", ty="title"), t("b", True)])) == 2        # previous not text
    assert len(_merge_prev([t("b", True)])) == 1                            # nothing to merge into
    assert len(_merge_prev([t("a"), t("b")])) == 2                          # flag unset


def test_simple_markdown_and_document_exports(tmp_path):
    bl = [Block("title", [0, 0, 1, .1], "T"), Block("text", [0, .1, 1, .2], "body"), Block("header", [0, 0, 1, .02], "H"),
          Block("equation", [0, .3, 1, .4], "x=1"), Block("code", [0, .5, 1, .6], "print(1)"),
          Block("image", [0, .6, 1, .7]), Block("table_caption", [0, .7, 1, .8], "cap")]
    d = Document(pages=[Page(0, 10, 10, (10, 10), bl)], backend="x", source="s.pdf")
    md = simple_md(d)
    assert md.splitlines()[0] == "# T" and "H" not in md and "$$\nx=1\n$$" in md and "```\nprint(1)\n```" in md
    assert "![image]()" in md and "*cap*" in md
    assert "H" in simple_md(d, drop_types=set())
    cl = d.to_content_list()
    assert len(cl) == 7 and cl[0]["page"] == 0
    assert json.loads(json.dumps(d.to_dict()))["pages"][0]["blocks"][0]["type"] == "title"
    paths = d.save(tmp_path, "name", style="legacy")
    assert paths["md"].name == "name.md" and paths["middle_json"].name == "name_middle.json"
    assert d.to_simple_markdown() == md
    with pytest.raises(ValueError):
        d.to_markdown("bogus")


# ---------------------------------------------------------------- official adaptation
def B(t, c=None, **extra):
    return Block(t, [0.1, 0.1, 0.5, 0.2], content=c, extra=extra)


def test_block_to_raw_mappings():
    f = official.block_to_raw
    assert f(B("equation_block", "x"), "teleocr") is None
    assert f(B("list", ""), "teleocr") is None and f(B("list", "- a"), "teleocr")["type"] == "list"
    assert f(B("seal", "s"), "teleocr")["type"] == "image" and f(B("seal", "s"), "teleocr")["sub_type"] == "seal"
    assert f(B("char", "c"), "teleocr")["type"] == "chart"
    assert f(B("algorithm", "a"), "teleocr")["type"] == "code"
    assert f(B("code", "<_python_>print(1)"), "teleocr")["content"] == "print(1)"
    assert f(B("text", "a $x$ b"), "teleocr")["content"] == "a \\(x\\) b"
    assert f(B("equation", "$$a$$"), "teleocr")["content"] == "a"
    # mineru passes through untouched, carrying its extra fields
    r = f(B("seal", "s", sub_type="seal", score=0.9), "mineru")
    assert r["type"] == "seal" and r["sub_type"] == "seal" and r["score"] == 0.9
    assert f(B("equation", "$$a$$"), "mineru")["content"] == "$$a$$"
    assert f(Block("text", [0, 0, 1, 1], "t", angle=None), "mineru")["angle"] == 0


def test_resolve_style():
    assert official.resolve_style("v4") == "v4" and official.resolve_style("legacy") == "legacy"
    assert official.resolve_style("auto") == "v4"
    with pytest.raises(ValueError):
        official.resolve_style("x")


@respx.mock
async def test_title_classifier_and_v4_types():
    layout = teleocr_layout(("100 50 900 100", "title"), ("100 120 900 170", "title"), ("100 200 900 300", "text"),
                            ("100 320 900 400", "ref_text"), ("100 420 900 440", "page_number"),
                            ("100 450 900 470", "header"), ("100 500 900 600", "algorithm"))
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=lambda r: chat(
        layout if TELEOCR_LAYOUT_PROMPT in prompt_of(r) else "content"))

    def cls(block, page_index):
        return ("doc_title", 1) if block["bbox"][1] < 0.1 else ("paragraph_title", 2)

    async with AsyncDocParser("teleocr", config=tele_cfg(), title_classifier=cls) as p:
        d = await p.parse(page_img())
    raw_types = [b["type"] for b in d.to_model_json()["pages"][0]]
    assert raw_types[:2] == ["doc_title", "paragraph_title"] and "code" in raw_types
    md = d.to_markdown("v4")
    assert md.startswith("# content") and "\n## content" in md
    mid = d.to_middle_json("v4")
    assert {b["type"] for b in mid["pages"][0]["blocks"]} >= {"doc_title", "paragraph_title", "text", "ref_text", "page_number", "header", "code"}


@respx.mock
async def test_partial_flag_and_page_index_map(tmp_path):
    pdf = tmp_path / "a.pdf"
    page_img(300, 300).save(pdf, "PDF", save_all=True, append_images=[page_img(300, 300) for _ in range(3)])
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=lambda r: chat(
        teleocr_layout(("100 100 900 900", "text")) if TELEOCR_LAYOUT_PROMPT in prompt_of(r) else "t"))
    async with AsyncDocParser("teleocr", config=tele_cfg(), dpi=72) as p:
        full = await p.parse(pdf)
        part = await p.parse(pdf, pages=[1, 3])
        img = await p.parse(page_img())
    assert not full.partial and part.partial and not img.partial
    assert full.to_model_json()["page_index_map"] == [] and part.to_model_json()["page_index_map"] == [1, 3]
    assert full.to_middle_json("v4")["is_full_document"] is True and part.to_middle_json("v4")["is_full_document"] is False


@respx.mock
async def test_v4_without_raw_blocks_raises():
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=lambda r: chat(
        teleocr_layout(("100 100 900 900", "text")) if TELEOCR_LAYOUT_PROMPT in prompt_of(r) else "t"))
    async with AsyncDocParser("teleocr", config=tele_cfg(), crop_assets=False) as p:
        d = await p.parse(page_img())
    assert d.pages[0].raw is None and d.pages[0].assets == {}
    with pytest.raises(ValueError, match="raw blocks"):
        d.to_markdown("v4")
    assert "t" in d.to_markdown("legacy")


# ---------------------------------------------------------------- sync facade + CLI
@respx.mock
def test_sync_facade_repeated_calls_and_running_loop_guard():
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=lambda r: chat(
        teleocr_layout(("100 100 900 900", "text")) if TELEOCR_LAYOUT_PROMPT in prompt_of(r) else "t"))
    p = DocParser("teleocr", config=tele_cfg())
    assert p.parse(page_img()).pages[0].blocks and p.parse(page_img()).pages[0].blocks        # reuse across loops
    assert len(p.parse_many([page_img(), page_img()])) == 2

    async def inside():
        return p.parse(page_img())

    with pytest.raises(RuntimeError):
        asyncio.run(inside())


@respx.mock
def test_cli(tmp_path, capsys):
    from docparse.cli import main
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=lambda r: chat(
        teleocr_layout(("100 100 900 300", "title"), ("100 400 900 600", "image")) if TELEOCR_LAYOUT_PROMPT in prompt_of(r) else "T"))
    png = tmp_path / "in.png"
    page_img().save(png)
    out = tmp_path / "out"
    assert main([str(png), "-b", "teleocr", "--base-url", BASE, "--model", "teleocr", "-o", str(out), "--style", "v4"]) == 0
    d = out / "in"
    for f in ("markdown.md", "middle_json.json", "structured_content.json", "model_output.json", "in_content_list.json"):
        assert (d / f).exists(), f
    assert any((d / "images").glob("*.jpg"))
    out2 = tmp_path / "out2"
    assert main([str(png), "--base-url", BASE, "--model", "teleocr", "-o", str(out2), "--style", "legacy", "--pages", "0"]) == 0
    assert (out2 / "in" / "in.md").exists() and (out2 / "in" / "in_middle.json").exists()


@respx.mock
async def test_teleocr_retries_transport_errors_and_5xx():
    calls = {"n": 0}

    def h(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadError("conn reset", request=req)
        if calls["n"] == 2:
            return chat("", status=503)
        return chat(teleocr_layout(("100 100 900 900", "text")) if TELEOCR_LAYOUT_PROMPT in prompt_of(req) else "ok")

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=h)
    async with AsyncDocParser("teleocr", config=BackendConfig(base_url=BASE, model="m", max_retries=3)) as p:
        d = await p.parse(page_img())
    assert d.pages[0].blocks[0].content == "ok" and calls["n"] == 4        # 2 failures + layout + text


@respx.mock
async def test_teleocr_gives_up_after_max_retries_and_does_not_retry_4xx():
    calls = {"n": 0}
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=lambda r: (calls.__setitem__("n", calls["n"] + 1), chat("", 503))[1])
    async with AsyncDocParser("teleocr", config=BackendConfig(base_url=BASE, model="m", max_retries=2)) as p:
        with pytest.raises(httpx.HTTPStatusError):
            await p.parse(page_img())
    assert calls["n"] == 3
    calls["n"] = 0
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=lambda r: (calls.__setitem__("n", calls["n"] + 1), httpx.Response(400, json={}))[1])
    async with AsyncDocParser("teleocr", config=BackendConfig(base_url=BASE, model="m", max_retries=2)) as p:
        with pytest.raises(httpx.HTTPStatusError):
            await p.parse(page_img())
    assert calls["n"] == 1


def test_pdf_with_junk_before_header(tmp_path):
    pdf = tmp_path / "a.pdf"
    page_img(300, 300).save(pdf, "PDF")
    junk = tmp_path / "junk.pdf"
    junk.write_bytes(b"my file,don't open it illegally.\r\n" + pdf.read_bytes())
    assert len(load_pages(junk, dpi=72)) == 1                       # pdfium reads it; must not be taken for an image
    with pytest.raises(Exception):                                  # a real non-PDF, non-image is still an error
        load_pages(b"hello world, not a document")


def test_degenerate_layout_boxes_dropped():
    bl = parse_layout_output("<box:513 0 545 0><label:page_number><up>\n<box:10 10 10 50><label:text><up>\n"
                             "<box:10 10 20 20 30 30 40 40><label:text><up>\n<box:10 10 50 50><label:text><up>")
    assert [(b.type) for b in bl] == ["text", "text"]               # zero-height, zero-width dropped; real ones kept


def test_empty_content_blocks_not_sent_to_upstream_validation():
    f = official.block_to_raw
    for t in ("page_number", "header", "text", "title", "footer", "image_caption", "equation"):
        assert f(B(t, None), "teleocr") is None and f(B(t, "  \n"), "mineru") is None
    assert f(B("image", None), "mineru") is not None and f(B("table", None), "teleocr") is not None


@respx.mock
async def test_blank_blocks_do_not_break_v4():
    layout = teleocr_layout(("100 100 900 200", "title"), ("100 300 900 400", "text"), ("100 420 900 440", "page_number"))
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=lambda r: chat(
        layout if TELEOCR_LAYOUT_PROMPT in prompt_of(r) else "   "))        # model returns blanks
    async with AsyncDocParser("teleocr", config=tele_cfg()) as p:
        d = await p.parse(page_img())
    assert d.to_markdown("v4") == "" and d.to_structured_content() is not None
