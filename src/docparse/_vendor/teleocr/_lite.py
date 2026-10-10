"""Dependency-free replacements for upstream helpers that pull in heavy packages.

- detect_lang: upstream uses fast_langdetect (fasttext model download); only the zh/ja/ko vs other
  distinction matters for markdown line-joining, so a script-ratio heuristic is enough.
- guess_language_by_text: upstream uses magika and has an unbound-variable bug on unprefixed code;
  models emit code as "<_lang_>source", which is all we need to parse.
"""
from __future__ import annotations


def detect_lang(text: str, max_len: int = 128) -> str:
    text = "".join(c for c in text.replace("\n", "")[:max_len] if not 0xD800 <= ord(c) <= 0xDFFF)
    if not text:
        return ""
    han = kana = hangul = letters = 0
    for c in text:
        o = ord(c)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            han += 1
        elif 0x3040 <= o <= 0x30FF:
            kana += 1
        elif 0xAC00 <= o <= 0xD7AF:
            hangul += 1
        elif c.isalpha():
            letters += 1
    total = han + kana + hangul + letters
    if not total:
        return ""
    if kana and kana / total > 0.05:
        return "ja"
    if hangul / total > 0.3:
        return "ko"
    if han / total > 0.3:
        return "zh"
    return "en"


def _code_clean(content: str | None) -> str:
    if not content:
        return ""
    lines = content.splitlines()
    start, end = 0, len(lines)
    if lines and lines[0].startswith("```"):
        start = 1
    if lines and end > start and lines[end - 1].strip() == "```":
        end -= 1
    return "\n".join(lines[start:end]).strip() if start < end else ""


def guess_language_by_text(code: str | None) -> tuple[str, str]:
    code = _code_clean(code)
    if code.startswith("<_"):
        end = code.find("_>")
        if end != -1:
            lang = code[2:end].lower()
            return (lang if lang != "unknown" else "txt"), code[end + 2:]
    return "txt", code
