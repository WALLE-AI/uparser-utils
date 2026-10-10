"""Backend-agnostic result types."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Block:
    type: str
    bbox: list[float]            # normalized [x1, y1, x2, y2] in 0-1 (page-relative)
    content: str | None = None
    angle: int | None = None
    page: int = 0
    merge_prev: bool = False     # MinerU: text block continues the previous text block
    image_path: str | None = None  # relative path of a cropped asset, key into Page.assets
    extra: dict[str, Any] = field(default_factory=dict)  # backend-specific raw fields (sub_type, cell_merge, score)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Page:
    index: int
    width: int
    height: int
    page_size: tuple[int, int] | None = None   # PDF points (px for image input); coordinate space of middle.json
    blocks: list[Block] = field(default_factory=list)
    assets: dict[str, bytes] = field(default_factory=dict)   # image_path -> JPEG bytes (legacy style)
    raw: list[dict[str, Any]] | None = None   # MinerU 4.0 model_list for this page (with image_base64 crops)


@dataclass
class Document:
    pages: list[Page]
    backend: str = ""
    source: str = ""
    partial: bool = False   # True when only a page subset was parsed (ModelJson.page_index_map)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # ---- output styles -----------------------------------------------------------------
    # style="v4"     MinerU 4.0 pipeline (docvortex): needs `pip install "docparse-sdk[official]"`
    # style="legacy" classic MinerU 2.x/3.x middle.json (para_blocks) via vendored MagicModel
    # style="auto"   v4 if docvortex is importable, else legacy (with a warning)

    def _style(self, style: str) -> str:
        from . import official

        return official.resolve_style(style)

    def to_markdown(self, style: str = "auto", **kwargs: Any) -> str:
        if self._style(style) == "v4":
            from . import official

            return official.to_markdown(self, **kwargs)
        from .middle_json import to_markdown

        return to_markdown(self, kwargs.get("img_dir", "images"))

    def to_middle_json(self, style: str = "auto") -> Any:
        """v4 -> dict in the docvortex MiddleJson schema (pages/blocks); legacy -> {"pdf_info": ...}."""
        if self._style(style) == "v4":
            from . import official

            return official.to_middle_json_dict(self)
        from .middle_json import to_middle_json

        return to_middle_json(self)

    def to_model_json(self) -> dict[str, Any]:
        """MinerU 4.0 ModelJson (raw, normalized model_list) -- the `model_output.json` of an official bundle."""
        from . import official

        return official.build_model_json(self).to_dict(skip_defaults=False)

    def to_structured_content(self) -> dict[str, Any]:
        from . import official

        return official.to_structured_content(self)

    def to_simple_markdown(self, **kwargs: Any) -> str:
        """Plain in-order rendering without block grouping (debug / fallback)."""
        from .render import to_markdown

        return to_markdown(self, **kwargs)

    def assets(self) -> dict[str, bytes]:
        return {k: v for p in self.pages for k, v in p.assets.items()}

    def save(self, out_dir: "str | Path", name: str | None = None, *, style: str = "auto",
             with_assets: bool = True) -> dict[str, "Path"]:
        """v4: markdown.md, middle_json.json, structured_content.json, model_output.json + image files
        (same bundle as MinerU 4.0's ParseResult.save). legacy: <name>.md, <name>_middle.json, images/."""
        out = Path(out_dir)
        if self._style(style) == "v4":
            from . import official

            return official.save_bundle(self, out, with_assets=with_assets)
        out.mkdir(parents=True, exist_ok=True)
        name = name or (Path(self.source).stem if self.source else "document")
        paths = {"md": out / f"{name}.md", "middle_json": out / f"{name}_middle.json"}
        paths["md"].write_text(self.to_markdown("legacy"), encoding="utf-8")
        paths["middle_json"].write_text(
            json.dumps(self.to_middle_json("legacy"), ensure_ascii=False, indent=2), encoding="utf-8")
        if with_assets:
            for rel, data in self.assets().items():
                f = out / "images" / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(data)
        return paths

    def to_content_list(self) -> list[dict[str, Any]]:
        """Flat list of blocks in reading order, each tagged with its page index."""
        return [b.to_dict() for p in self.pages for b in p.blocks]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "source": self.source,
            "pages": [
                {"index": p.index, "width": p.width, "height": p.height,
                 "blocks": [b.to_dict() for b in p.blocks]}
                for p in self.pages
            ],
        }
