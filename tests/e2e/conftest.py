"""Live end-to-end tests: need the vLLM services from deploy_ocr_vlm.sh (skipped when unreachable).

Env overrides: DOCPARSE_MINERU_URL, DOCPARSE_TELEOCR_URL, DOCPARSE_E2E_PAPER, DOCPARSE_E2E_SPEC.
"""
import os

import httpx
import pytest

URLS = {
    "mineru": os.getenv("DOCPARSE_MINERU_URL", "http://127.0.0.1:8003"),
    "teleocr": os.getenv("DOCPARSE_TELEOCR_URL", "http://127.0.0.1:8004"),
}
DATA = "/home/dataset-s3-0/datasets/PDFs/data/"
# paper: equations (p2), figures (p0,p7), tables (p3), references/lists (p9), headers/footers everywhere
PAPER = os.getenv("DOCPARSE_E2E_PAPER", DATA + "04.2 论文 基于水质目标可达性分析的城市内河生态补水方案 第一.pdf")
SPEC = os.getenv("DOCPARSE_E2E_SPEC", DATA + "GBT51121-2015 风力发电工程施工与验收规范.pdf")


def _up(url):
    try:
        return httpx.get(url + "/v1/models", timeout=3).status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session", params=["mineru", "teleocr"])
def backend(request):
    if not _up(URLS[request.param]):
        pytest.skip(f"{request.param} service not reachable at {URLS[request.param]}")
    if not os.path.exists(PAPER):
        pytest.skip(f"missing sample pdf {PAPER}")
    return request.param


@pytest.fixture(scope="session")
def paper_doc(backend):
    from docparse import DocParser

    return DocParser(backend, base_url=URLS[backend], concurrency=32).parse(PAPER, pages=[0, 2, 3, 7, 9])
