# Output formats

`<out>/<input-stem>/` always gets `<stem>_content_list.json` (flat blocks: `type`, `bbox` 0-1, `content`, `page`).

## style v4 (MinerU 4.0 bundle, needs docvortex)

| File | Content |
|---|---|
| `markdown.md` | rendered Markdown, tables as HTML, `images/...` references |
| `middle_json.json` | top-level keys `metadata`, `extensions`, `pages`, `schema` (NOT `pdf_info`) |
| `structured_content.json` | structured blocks |
| `model_output.json` | raw model list per page |
| `images/*.jpg` | cropped figures / tables / equations, e.g. `page_6_table_5.jpg` |

## style legacy (classic MinerU 2.x/3.x)

| File | Content |
|---|---|
| `<stem>.md` | Markdown, tables as HTML |
| `<stem>_middle.json` | `{"pdf_info": [ one entry per page ]}`, coordinates in PDF points |
| `images/` | cropped assets |

## Block types seen in practice

text, title, list, table, table_caption, image, image_caption, image_footnote, equation, header, footer,
page_number, aside_text. `header`/`footer`/`page_number` are not rendered into the Markdown.

## v4 limits (need MinerU's local GPU models; not reproduced)

- every `title` becomes `paragraph_title` level 2 (override with `DocParser(..., title_classifier=fn)`)
- each text block has a single line = its bbox (log shows `bbox_fallback=N` warnings, harmless; `dropped=0`
  means nothing was lost)
- no native PDF text-layer filling, local formula recognition, seal OCR or cross-page table merge
- tables with embedded pictures are split into `table_image` crops; compare against the legacy Markdown if a
  table looks incomplete in v4
