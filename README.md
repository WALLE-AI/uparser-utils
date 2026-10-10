# docparse-sdk

Unified Python client for the **MinerU2.5-Pro** and **TeleOCR** document-parsing VLM services
(vLLM, OpenAI-compatible). Pure client: no torch / vllm in the install.

```
pip install "docparse-sdk[all]"         # MinerU + TeleOCR backends + MinerU 4.0 output (docvortex)
pip install "docparse-sdk[mineru]"      # MinerU + TeleOCR backends, legacy output
pip install docparse-sdk                # TeleOCR backend only
```

## Usage

```python
from docparse import DocParser

parser = DocParser("teleocr", base_url="http://127.0.0.1:8004", model="teleocr", concurrency=16)
doc = parser.parse("a.pdf", pages=[0, 1])
print(doc.to_markdown())
doc.to_content_list()   # flat blocks: type / bbox (0-1) / content / page
doc.to_middle_json()    # MinerU-style middle.json (grouped para_blocks, PDF-point coordinates)
doc.save("out/")        # out/<name>.md + <name>_middle.json + images/ (cropped figures/seals/charts)
```

```python
from docparse import AsyncDocParser

async with AsyncDocParser("mineru", base_url="http://127.0.0.1:8003") as p:
    docs = await p.parse_many(["a.pdf", "b.png"])
```

CLI: `docparse a.pdf -b mineru --base-url http://127.0.0.1:8003 -o out/`

## How it works

| | MinerU | TeleOCR |
|---|---|---|
| transport | `mineru-vl-utils` `MinerUClient(http-client)` | httpx (`backends/teleocr.py`) |
| layout parse / crop | mineru-vl-utils | ported from `TeleOCR_client.py` |
| post-process | mineru-vl-utils | vendored `_vendor/teleocr/post_process` |

Both backends run: render page -> layout detection -> crop each block -> concurrent per-block
recognition -> post-process -> unified `Block` list. 
### Output styles

```python
doc.save("out/", style="v4")        # MinerU 4.0 bundle: markdown.md, middle_json.json,
                                    #   structured_content.json, model_output.json, images/*.jpg
doc.to_markdown("v4"); doc.to_middle_json("v4"); doc.to_structured_content(); doc.to_model_json()
doc.save("out/", style="legacy")    # classic <name>.md + <name>_middle.json ({"pdf_info": [...]})
```
`style="auto"` (default) uses v4 when `docvortex` is installed (`pip install "docparse-sdk[official]"`),
otherwise legacy.

**v4 = MinerU 4.0.11 `xhigh` post-processing**, built on the same code upstream runs:

```
VLM blocks -> convert/normalize_xhigh -> title typing -> image_base64 crops -> _normalize_pdf_model_list
  -> ModelJson -> docvortex.model_json_to_middle_json -> docvortex.render (markdown / structured_content)
```
The normalization step is vendored from `mineru` (`_vendor/mineru4`); the middle-json build, asset
materialization and renderers are `docvortex` itself, so markdown / middle_json match upstream for
identical model_list input.

**Not reproduced** (they need MinerU's local GPU models, outside a VLM-service client):
- layout small model: `title` is typed `paragraph_title` level 2 for all titles (upstream makes some `doc_title`
  level 1). Override with `DocParser(..., title_classifier=fn)`.
- OCR detection (line boxes): each text block gets one line = its bbox.
- native PDF text-layer filling for digital PDFs, local formula recognition, seal OCR, cross-page table merge,
  optional LLM-aided title leveling.

TeleOCR output is adapted into the same model_list (seal->image/seal, char->chart, algorithm->code,
`$$..$$` stripped from equations, un-extracted `list` regions dropped); TeleOCR has no upstream docvortex
path, so for it v4 means "MinerU 4.0 renderers over TeleOCR content".

#### Legacy style
backend blocks -> MagicModel -> `pdf_info` middle.json -> `union_make` (classic MinerU 2.x/3.x; vendored
from TeleOCR). mineru-vl-utils debug logs are muted; re-enable with `logger.enable("mineru_vl_utils")`.

Notes: TeleOCR's upstream `no_repeat_ngram_size=100` needs a server-side logits processor, so it is
off by default (`TeleOCRBackend(..., no_repeat_ngram_size=100)` to enable). Default MinerU
`max-model-len` of the deployed service is 8192; very large crops may exceed it.

## Build / publish

```
python -m build && twine check dist/*
twine upload --repository testpypi dist/*     # or your private index
```
See NOTICE.md for third-party license notes (resolve the TeleOCR license before a public release).

## Testing

```
pytest                              # offline unit tests (fake vLLM servers via respx), ~35 tests
pytest tests/e2e                    # live end-to-end vs MinerU:8003 / TeleOCR:8004 (auto-skipped if down)
pytest -m slow tests/e2e/test_full.py   # whole 73-page document, concurrent stress, 40-file corpus robustness
pytest -m "slow or not slow" --cov=docparse   # everything (~13 min)
```
Env: `DOCPARSE_MINERU_URL`, `DOCPARSE_TELEOCR_URL`, `DOCPARSE_E2E_PAPER`, `DOCPARSE_E2E_SPEC`, `DOCPARSE_CORPUS_N`.
