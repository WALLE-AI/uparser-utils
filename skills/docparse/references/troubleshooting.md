# Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ServerError: Failed to get model name from http://HOST:PORT ... 404 nginx` (MinerU, URL has a path prefix) | Old SDK build dropped the `/mineru` prefix (mineru-vl-utils keeps a path only if `server_url` ends in `/`). Use an SDK version whose `backends/mineru.py` has `_keep_path_prefix`. |
| `Model '...' not found in ... /v1/models` | Wrong `model=`. `curl <base_url>/v1/models` and copy the `id`. |
| Python process dies with exit code 139 / "access violation" while importing `onnxruntime` (Windows, v4 style) | `onnxruntime` 1.31 and 1.22 fail here; `pip install "onnxruntime==1.20.1"` works. Or use `--style legacy`, which does not need docvortex. |
| `ModuleNotFoundError: docparse` | `pip install "docparse-sdk[all]"` (repo: `pip install -e ".[all]"`). |
| pip warns `mineru requires mineru-vl-utils<1` | The `mineru` package pins an older `mineru-vl-utils`. Create a clean venv (`uv venv; uv pip install -e ".[all]"`). |
| `TypeError ... unexpected keyword argument 'timeout'` | Extra kwargs to `DocParser(...)` go to the MinerU client. Timeouts live in `BackendConfig(timeout=...)` (default 600 s). |
| Timeouts, 429, 5xx under load | Lower `-c`; `BackendConfig.max_retries` defaults to 3 with backoff. |
| Very large crops fail | Server `max-model-len` is 8192; oversized crops can exceed it. Retry that page at lower dpi (`DocParser(dpi=...)`, default 150). |
| v4 Markdown has fewer tables than expected | See `output-formats.md`; compare with `--style legacy`. |
| Cover-page text missing in TeleOCR Markdown | TeleOCR classes it as `header`; read `*_content_list.json` or switch to mineru. |
