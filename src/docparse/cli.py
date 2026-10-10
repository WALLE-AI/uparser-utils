from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import BackendConfig
from .parser import DocParser


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="docparse", description="Parse PDF/images with MinerU or TeleOCR services.")
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("-b", "--backend", choices=["mineru", "teleocr"], default="teleocr")
    ap.add_argument("--base-url", default=None, help="default: 8003 for mineru, 8004 for teleocr")
    ap.add_argument("--model", default=None)
    ap.add_argument("-c", "--concurrency", type=int, default=16)
    ap.add_argument("--pages", default=None, help="0-based, comma separated, e.g. 0,1,5")
    ap.add_argument("--style", choices=["auto", "v4", "legacy"], default="auto",
                    help="v4: MinerU 4.0 bundle (needs docparse-sdk[official]); legacy: classic middle.json")
    ap.add_argument("-o", "--output", type=Path, default=Path("docparse_out"))
    args = ap.parse_args(argv)

    cfg = None
    if args.base_url:
        cfg = BackendConfig(base_url=args.base_url, model=args.model, concurrency=args.concurrency)
    parser = DocParser(args.backend, config=cfg, concurrency=args.concurrency)
    pages = [int(x) for x in args.pages.split(",")] if args.pages else None

    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.inputs:
        doc = parser.parse(path, pages=pages)
        out = args.output / path.stem
        paths = doc.save(out, path.stem, style=args.style)
        (out / f"{path.stem}_content_list.json").write_text(
            json.dumps(doc.to_content_list(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path} -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
