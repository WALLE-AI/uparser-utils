"""Heavy live tests (opt-in): `pytest tests/e2e/test_full.py -m slow`  (or run with -m "slow or not slow")."""
import collections
import glob
import json
import os
import random
import re
import time
import traceback

import pytest

from docparse import AsyncDocParser, DocParser
from .conftest import DATA, SPEC, URLS

pytest.importorskip("docvortex")
from docvortex.schema import MiddleJson  # noqa: E402

pytestmark = pytest.mark.slow


def test_whole_document(backend, tmp_path):
    """All 73 pages of the spec in one call: no page lost, order kept, both styles + bundle written."""
    p = DocParser(backend, base_url=URLS[backend], concurrency=64)
    t = time.time()
    doc = p.parse(SPEC)
    dt = time.time() - t
    n = len(doc.pages)
    nblocks = sum(len(pg.blocks) for pg in doc.pages)
    print(f"\n[{backend}] {n} pages, {nblocks} blocks in {dt:.1f}s ({n/dt:.2f} page/s)")
    assert n == 73 and [pg.index for pg in doc.pages] == list(range(73)) and not doc.partial
    empty = [pg.index for pg in doc.pages if not pg.blocks]
    assert len(empty) <= 3, f"too many empty pages: {empty}"                  # cover/blank pages tolerated
    assert sum(1 for pg in doc.pages for b in pg.blocks if b.type == "page_number") >= 60
    paths = doc.save(tmp_path / "v4", style="v4")
    mid = json.loads((tmp_path / "v4" / "middle_json.json").read_text(encoding="utf-8"))
    assert MiddleJson.from_dict(mid) and mid["is_full_document"] is True and len(mid["pages"]) == 73
    md = (tmp_path / "v4" / "markdown.md").read_text(encoding="utf-8")
    body = sum(len(re.sub(r"\s", "", b.content or "")) for pg in doc.pages for b in pg.blocks
               if b.type in ("text", "title", "list", "ref_text"))
    squash = lambda t: len(re.sub(r"\s", "", re.sub(r"<[^>]+>|\$|\\\(|\\\)", "", t)))
    assert body > 20_000 and squash(md) >= 0.9 * body, (body, squash(md))      # nothing silently dropped
    assert "总则" in md
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)
    assert all((tmp_path / "v4" / r).exists() for r in refs)
    legacy = doc.to_middle_json("legacy")
    assert len(legacy["pdf_info"]) == 73
    assert squash(doc.to_markdown("legacy")) >= 0.9 * body


def test_stress_concurrent_documents(backend):
    """8 documents x 6 pages in flight at once through one parser; results stay independent and ordered."""
    import asyncio

    async def run():
        async with AsyncDocParser(backend, base_url=URLS[backend], concurrency=96) as p:
            t = time.time()
            docs = await p.parse_many([SPEC] * 8, pages=list(range(10, 16)))
            return docs, time.time() - t

    docs, dt = asyncio.run(run())
    print(f"\n[{backend}] 48 pages / 8 docs in {dt:.1f}s ({48/dt:.2f} page/s)")
    assert all([pg.index for pg in d.pages] == list(range(10, 16)) for d in docs)
    sig = lambda d: [len(pg.blocks) for pg in d.pages]
    base = sig(docs[0])
    assert all(abs(a - b) <= 2 for d in docs for a, b in zip(sig(d), base))      # greedy decoding => ~deterministic
    assert all(len(d.to_markdown("v4")) > 1000 for d in docs)


def test_corpus_robustness(backend, tmp_path):
    """Random sample of the PDF corpus (first pages): nothing may raise; every output style must build."""
    n = int(os.getenv("DOCPARSE_CORPUS_N", "24"))
    files = sorted(glob.glob(DATA + "*.pdf"))
    random.Random(20261010).shuffle(files)
    p = DocParser(backend, base_url=URLS[backend], concurrency=48)
    failures, seen_types = [], collections.Counter()
    done = 0
    for f in files:
        if done >= n:
            break
        try:
            doc = p.parse(f, pages=[0, 1, 2])
        except Exception as e:                                   # unreadable / encrypted source is not our bug
            msg = str(e)
            if "PdfiumError" in type(e).__name__ or "Failed to load document" in msg:
                continue
            failures.append((os.path.basename(f), "parse", f"{type(e).__name__}: {msg[:200]}"))
            done += 1
            continue
        done += 1
        for pg in doc.pages:
            seen_types.update(b.type for b in pg.blocks)
        for step, fn in {
            "v4-md": lambda: doc.to_markdown("v4"), "v4-bundle": lambda: doc.save(tmp_path / f"{done}", style="v4"),
            "legacy-md": lambda: doc.to_markdown("legacy"), "legacy-mid": lambda: doc.to_middle_json("legacy"),
            "structured": doc.to_structured_content,
        }.items():
            try:
                fn()
            except Exception as e:
                failures.append((os.path.basename(f), step, f"{type(e).__name__}: {str(e)[:300]}"))
    print(f"\n[{backend}] corpus {done} files; block types seen: {dict(seen_types)}")
    for f in failures:
        print("  FAIL", f)
    assert not failures
