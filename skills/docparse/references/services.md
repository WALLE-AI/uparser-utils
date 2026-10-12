# Known services

Both are vLLM OpenAI-compatible servers behind one nginx gateway. A path prefix is part of `base_url`;
the SDK appends `/v1/...` itself.

| Backend | base_url | model id | max_model_len |
|---|---|---|---|
| mineru | `http://36.103.239.202:9005/mineru` | `mineru2.5-pro-1.2b` | 8192 |
| teleocr | `http://36.103.239.202:9005/teleocr` | `teleocr` | 8192 |

Verify before use: `curl -s <base_url>/v1/models` should return JSON with the model id.

Local default ports (from the deployment script): MinerU `http://127.0.0.1:8003`, TeleOCR `http://127.0.0.1:8004`.

Environment variables read by `scripts/parse_document.py`:
`DOCPARSE_MINERU_URL`, `DOCPARSE_TELEOCR_URL`, `DOCPARSE_BASE_URL` (fallback for either).

These endpoints are internal and unauthenticated as of this writing; if the gateway later requires a key,
use `BackendConfig(api_key=...)`.
