#!/usr/bin/env python3
"""Parse PDFs / images into Markdown + middle.json with the docparse SDK (MinerU or TeleOCR service).

Pages are 1-based here (--pages 1-5,8). Prints a Markdown summary to stdout (--format json for JSON); logs go to stderr.

Service URL resolution: --base-url > $DOCPARSE_<BACKEND>_URL > $DOCPARSE_BASE_URL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

DEFAULT_MODEL = {"mineru": "mineru2.5-pro-1.2b", "teleocr": "teleocr"}


def parse_pages(spec: str | None) -> list[int] | None:
    """'1-3,7' (1-based) -> [0, 1, 2, 6]."""
    if not spec:
        return None
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        a, b = int(lo), int(hi or lo)
        if a < 1 or b < a:
            raise SystemExit(f"bad --pages segment {part!r}: pages are 1-based and ranges ascend")
        pages.extend(range(a - 1, b))
    return pages


def to_markdown(results: list[dict]) -> str:
    lines: list[str] = []
    for r in results:
        lines += [f"## {Path(r['input']).name}", "",
                  f"- 输出目录: `{r['output_dir']}`", f"- 解析页数: {r['pages_parsed']}",
                  f"- 图片文件数: {r['image_count']}", "", "| 块类型 | 数量 |", "|---|---|"]
        lines += [f"| {t} | {n} |" for t, n in sorted(r["blocks"].items(), key=lambda kv: -kv[1])]
        lines += ["", "输出文件:", ""] + [f"- `{f}`" for f in r["files"]] + [""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="PDF or image files")
    ap.add_argument("-b", "--backend", choices=["mineru", "teleocr"], default="mineru")
    ap.add_argument("--base-url", default=None, help="service root, a reverse-proxy path prefix is fine")
    ap.add_argument("--model", default=None)
    ap.add_argument("-c", "--concurrency", type=int, default=32)
    ap.add_argument("--pages", default=None, help="1-based, e.g. 1-5,8 (default: all pages)")
    ap.add_argument("--style", choices=["auto", "v4", "legacy"], default="auto")
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown", help="summary format on stdout")
    ap.add_argument("-o", "--output", type=Path, default=Path("docparse_out"))
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to GBK

    try:
        from docparse import DocParser
    except ImportError:
        print('docparse is not installed. Run: pip install "docparse-sdk[all]"  '
              '(or, from the repo: pip install -e ".[all]")', file=sys.stderr)
        return 2

    url = (args.base_url or os.getenv(f"DOCPARSE_{args.backend.upper()}_URL") or os.getenv("DOCPARSE_BASE_URL"))
    if not url:
        print(f"no service URL: pass --base-url or set DOCPARSE_{args.backend.upper()}_URL", file=sys.stderr)
        return 2
    missing = [str(p) for p in args.inputs if not p.is_file()]
    if missing:
        print("input not found: " + ", ".join(missing), file=sys.stderr)
        return 2

    parser = DocParser(args.backend, base_url=url, model=args.model or DEFAULT_MODEL[args.backend],
                       concurrency=args.concurrency)
    pages = parse_pages(args.pages)
    results = []
    for path in args.inputs:
        doc = parser.parse(path, pages=pages)
        out = args.output / path.stem
        doc.save(out, path.stem, style=args.style)
        blocks = doc.to_content_list()
        (out / f"{path.stem}_content_list.json").write_text(
            json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
        files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and "images" not in p.parts)
        results.append({
            "input": str(path), "output_dir": str(out), "pages_parsed": len({b.get("page") for b in blocks}),
            "blocks": dict(Counter(b["type"] for b in blocks)), "files": files,
            "image_count": sum(1 for p in out.rglob("*") if p.is_file() and "images" in p.parts),
        })
    print(json.dumps(results, ensure_ascii=False, indent=2) if args.format == "json" else to_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
