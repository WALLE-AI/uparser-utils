# Vendored from MinerU 4.0.11 (mineru/backend/analysis/pdf/layout.py); MinerU Open Source License
# (Apache-2.0 + additional terms, see NOTICE.md). Imports adapted for docparse-sdk.
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from .constants import VLM_MODEL_LIST_FIELDS, VLM_VISUAL_ANNOTATION_TYPE_MAP


def _convert_vlm_results_to_model_list(
    vlm_results: Iterable[Iterable[Mapping[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """将 VLM 结果投影为允许进入 Analyze 的字段，丢弃未采用的外部属性。"""
    return [
        [
            {field_name: deepcopy(value) for field_name, value in dict(block).items() if field_name in VLM_MODEL_LIST_FIELDS}
            for block in page_blocks
        ]
        for page_blocks in vlm_results
    ]


def _normalize_xhigh_vlm_blocks(model_list: list[list[dict[str, Any]]]) -> None:
    """归一化 xhigh VLM 的视觉注释类型。"""
    for page_model_list in model_list:
        for block in page_model_list:
            normalized_type = VLM_VISUAL_ANNOTATION_TYPE_MAP.get(block.get("type"))
            if normalized_type is not None:
                block["type"] = normalized_type
