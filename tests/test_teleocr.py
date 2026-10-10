import httpx
import pytest
import respx
from PIL import Image

from docparse import AsyncDocParser, BackendConfig
from docparse.backends.teleocr import parse_layout_output

LAYOUT = "<box:100 100 900 200><label:title><up>\n<box:100 300 900 500><label:text><up>"


def test_parse_layout():
    blocks = parse_layout_output(LAYOUT + "\ngarbage")
    assert [b.type for b in blocks] == ["title", "text"]
    assert blocks[0].bbox == [0.1, 0.1, 0.9, 0.2] and blocks[0].angle == 0


@respx.mock
async def test_end_to_end():
    def reply(request):
        import json
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"][-1]["text"]
        text = LAYOUT if "layout" in prompt else "Hello $x$"
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    respx.post("http://x/v1/chat/completions").mock(side_effect=reply)
    cfg = BackendConfig(base_url="http://x", model="teleocr")
    async with AsyncDocParser("teleocr", config=cfg) as p:
        doc = await p.parse(Image.new("RGB", (400, 600), "white"))
    assert len(doc.pages) == 1 and len(doc.pages[0].blocks) == 2
    assert doc.pages[0].blocks[1].content.startswith("Hello")
    assert doc.to_markdown("legacy").startswith("# Hello")


@respx.mock
async def test_middle_json_and_assets(tmp_path):
    layout = ("<box:100 100 900 200><label:title><up>\n<box:100 300 900 500><label:image><up>\n"
              "<box:100 520 900 560><label:image_caption><up>")

    def reply(request):
        import json
        prompt = json.loads(request.content)["messages"][-1]["content"][-1]["text"]
        return httpx.Response(200, json={"choices": [{"message": {"content": layout if "layout" in prompt else "Cap"}}]})

    respx.post("http://x/v1/chat/completions").mock(side_effect=reply)
    cfg = BackendConfig(base_url="http://x", model="teleocr")
    async with AsyncDocParser("teleocr", config=cfg) as p:
        doc = await p.parse(Image.new("RGB", (400, 600), "white"))
    mj = doc.to_middle_json("legacy")
    info = mj["pdf_info"][0]
    assert info["page_size"] == [400, 600] and info["page_idx"] == 0
    types = [b["type"] for b in info["para_blocks"]]
    assert types == ["title", "image"]                    # caption folded into its image block
    assert [b["type"] for b in info["para_blocks"][1]["blocks"]] == ["image_body", "image_caption"]
    md = doc.to_markdown("legacy")
    assert md.startswith("# Cap") and "![](images/image/" in md
    paths = doc.save(tmp_path, style="legacy")
    assert paths["middle_json"].exists() and any((tmp_path / "images" / "image").glob("*.jpg"))


@respx.mock
async def test_v4_official_bundle(tmp_path):
    pytest.importorskip("docvortex")
    layout = ("<box:100 100 900 200><label:title><up>\n<box:100 300 900 500><label:image><up>\n"
              "<box:100 520 900 560><label:image_caption><up>\n<box:100 600 900 700><label:text><up>")

    def reply(request):
        import json
        prompt = json.loads(request.content)["messages"][-1]["content"][-1]["text"]
        return httpx.Response(200, json={"choices": [{"message": {"content": layout if "layout" in prompt else "Cap"}}]})

    respx.post("http://x/v1/chat/completions").mock(side_effect=reply)
    cfg = BackendConfig(base_url="http://x", model="teleocr")
    async with AsyncDocParser("teleocr", config=cfg) as p:
        doc = await p.parse(Image.new("RGB", (400, 600), "white"))
    mj = doc.to_model_json()
    types = [b["type"] for b in mj["pages"][0]]
    assert types == ["paragraph_title", "image", "caption", "text"]      # raw types, caption folded later
    md = doc.to_markdown("v4")
    assert md.startswith("## Cap") and "![" in md
    paths = doc.save(tmp_path, style="v4")
    assert {"markdown.md", "middle_json.json", "structured_content.json", "model_output.json"} <= set(paths)
    assert any(k.startswith("images/") for k in paths)
    import json
    assert "image_base64" not in (tmp_path / "middle_json.json").read_text()


def test_teleocr_math_adaptation():
    from docparse.official import _teleocr_equation as eq, _teleocr_inline_math as inl

    assert eq("$$a+b$$") == "a+b"
    assert eq("$\\frac{a}{b}\\quad$ (1)") == "\\frac{a}{b} \\tag {1}"
    assert eq("$$ x=1 \\qquad $$ （2.3）") == "x=1 \\tag {2.3}"
    assert inl("面积 $m^{2}$ ，流量 $m^{3}/s$") == "面积 \\(m^{2}\\) ，流量 \\(m^{3}/s\\)"
    assert inl("price 5$ only") == "price 5$ only"            # unbalanced -> untouched
