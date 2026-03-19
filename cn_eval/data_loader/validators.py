"""
数据校验模块 — 检查加载的数据是否完整、一致。
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from .schema import Prompt, ModelOutput, ABMapping, AlignedPair

logger = logging.getLogger(__name__)


class DataValidator:
    """数据完整性和一致性校验。"""

    @staticmethod
    def validate_prompts(prompts: list[Prompt]) -> list[str]:
        """校验 Prompt 列表，返回警告信息列表。"""
        warnings: list[str] = []

        ids = [p.prompt_id for p in prompts]
        dupes = [pid for pid, cnt in Counter(ids).items() if cnt > 1]
        if dupes:
            warnings.append(f"存在重复 prompt_id: {dupes[:10]}")

        empty = [p.prompt_id for p in prompts if not p.text.strip()]
        if empty:
            warnings.append(f"{len(empty)} 条 prompt 文本为空: {empty[:5]}")

        for w in warnings:
            logger.warning("[Validator] %s", w)
        return warnings

    @staticmethod
    def validate_model_outputs(
        outputs: list[ModelOutput],
        prompt_ids: set[str] | None = None,
    ) -> list[str]:
        """校验模型输出。"""
        warnings: list[str] = []

        empty_resp = [o.prompt_id for o in outputs if not o.response.strip()]
        if empty_resp:
            warnings.append(f"{len(empty_resp)} 条模型输出 response 为空")

        if prompt_ids:
            output_ids = {o.prompt_id for o in outputs}
            missing = prompt_ids - output_ids
            extra = output_ids - prompt_ids
            if missing:
                warnings.append(f"{len(missing)} 个 prompt 缺少模型输出: {list(missing)[:5]}")
            if extra:
                warnings.append(f"{len(extra)} 条输出无对应 prompt: {list(extra)[:5]}")

        for w in warnings:
            logger.warning("[Validator] %s", w)
        return warnings

    @staticmethod
    def validate_alignment(
        pairs: list[AlignedPair],
    ) -> list[str]:
        """校验对齐后的数据。"""
        warnings: list[str] = []

        missing_baseline = [p.prompt_id for p in pairs if not p.baseline_response.strip()]
        missing_candidate = [p.prompt_id for p in pairs if not p.candidate_response.strip()]

        if missing_baseline:
            warnings.append(f"{len(missing_baseline)} 条缺少 baseline 回答")
        if missing_candidate:
            warnings.append(f"{len(missing_candidate)} 条缺少 candidate 回答")

        for w in warnings:
            logger.warning("[Validator] %s", w)
        return warnings

    @staticmethod
    def check_coverage(
        prompt_ids: set[str],
        version_outputs: dict[str, list[ModelOutput]],
    ) -> dict[str, Any]:
        """检查各版本对 prompt 的覆盖情况。"""
        report = {}
        for version, outputs in version_outputs.items():
            output_ids = {o.prompt_id for o in outputs}
            covered = prompt_ids & output_ids
            missing = prompt_ids - output_ids
            report[version] = {
                "total_prompts": len(prompt_ids),
                "covered": len(covered),
                "missing": len(missing),
                "coverage_rate": f"{len(covered) / len(prompt_ids) * 100:.1f}%",
                "missing_ids": sorted(missing)[:10],
            }
        return report
