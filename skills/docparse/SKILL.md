---
name: docparse
description: Parse PDFs and images (scans, papers, standards, spec documents, tables, formulas) into Markdown, middle.json and a flat content list by calling the MinerU2.5-Pro or TeleOCR VLM services through the docparse-sdk. Use this skill whenever the user wants to OCR a document, convert a PDF to Markdown, extract tables/formulas/figures from a PDF or image, produce MinerU-style middle.json, compare MinerU vs TeleOCR output, or mentions docparse, uparser, MinerU or TeleOCR - even if they do not name the tool.
compatibility: Python 3.10+, network access to a MinerU/TeleOCR vLLM service, pip install "docparse-sdk[all]"
---

# docparse

Pure-client SDK for two document-parsing VLM services (vLLM, OpenAI-compatible). It renders pages,
detects layout, crops each block, recognises blocks concurrently, post-processes, and writes Markdown +
middle.json. No GPU or torch needed locally.

## Quick start

1. Make sure the SDK is importable: `python -c "import docparse"`. If not: `pip install "docparse-sdk[all]"`
   (from the repo checkout: `pip install -e ".[all]"`). Use a fresh venv if the environment already has the
   `mineru` package, its pins conflict.
2. Check the service is up and pick a backend (see `references/services.md` for known endpoints):
   `curl -s <base-url>/v1/models`
3. Run the bundled script (paths relative to this skill folder):

```bash
python scripts/parse_document.py "path/to/file.pdf" -b mineru \
    --base-url http://HOST:PORT/mineru -c 32 --pages 1-5 -o out/
```

It prints a Markdown summary (block counts per type, output files); add `--format json` for JSON. Omit `--pages` to parse everything.
`--pages` is **1-based** in this script (the SDK's own `pages=[...]` and `docparse --pages` are 0-based).

## Choosing options

- **Backend**: `mineru` (default; best general layout, formulas, OTSL tables) or `teleocr` (keeps dot leaders,
  finds more figures/captions, but classes cover-page text as `header`, which Markdown drops). If quality
  matters, parse a few pages with both and compare before committing to a full run.
- **Concurrency**: `-c 32` is safe; 64 parsed 122 pages in ~135 s on the shared service. Lower it if you see
  429/5xx/timeouts.
- **Style**: `auto` writes the MinerU 4.0 bundle (`v4`) when `docvortex` is installed, otherwise `legacy`.
  Pass `--style legacy` for the classic `{"pdf_info": [...]}` middle.json. Output layouts differ, see
  `references/output-formats.md`.
- **Big documents**: do a 3-8 page trial first, check the Markdown, then run the full file in the background
  (a 122-page PDF takes a couple of minutes) and poll the output instead of blocking.

## Python API (when the script is not enough)

```python
from docparse import DocParser
parser = DocParser("mineru", base_url="http://HOST:PORT/mineru", model="mineru2.5-pro-1.2b", concurrency=32)
doc = parser.parse("a.pdf", pages=[0, 1])      # 0-based
doc.to_markdown(); doc.to_middle_json(); doc.to_content_list()
doc.save("out/", style="v4")
```
Async: `AsyncDocParser` + `await p.parse_many([...])`. Always pass `model=` explicitly for a gateway that
serves several models.

## After parsing

Report to the user: output directory, block counts, and anything suspicious. Do not claim the content is
accurate unless you read it; spot-check the Markdown (headings, one table, one formula) and say that is what
you checked. Known limits of the v4 style: title levels are all level 2, text blocks have one line per bbox,
no native PDF text-layer filling or cross-page table merge (details in `references/output-formats.md`).

## Troubleshooting

Read `references/troubleshooting.md` for: 404 on `/v1/models` through a path prefix, `Model ... not found`,
`onnxruntime` crashes on Windows, timeouts / max-model-len errors, dependency conflicts.
