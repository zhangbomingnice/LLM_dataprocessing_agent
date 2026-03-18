from __future__ import annotations

import json
import logging
from pathlib import Path

from config import LLMConfig
from utils.schema import OutputItem, EvalResult
from utils.io import write_jsonl, write_json
from utils.file_writer import write_output

from .base import BaseAgent

logger = logging.getLogger(__name__)


class AggregatorAgent(BaseAgent):
    """Agent 4: 整合所有处理结果，校验格式，打包输出。"""

    name = "Aggregator"

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config, system_prompt="")

    async def run(
        self,
        items: list[OutputItem],
        output_path: Path,
        report_path: Path | None = None,
    ) -> dict:
        """校验并输出最终结果。"""
        logger.info("[Aggregator] 开始整合 %d 条数据...", len(items))

        valid_items: list[OutputItem] = []
        errors: list[dict] = []

        for item in items:
            issues = self._validate(item)
            if issues:
                errors.append({"id": item.id, "issues": issues})
                logger.warning("[Aggregator] 条目 %s 校验问题: %s", item.id, issues)
            valid_items.append(item)

        write_output(valid_items, output_path)
        logger.info("[Aggregator] 已写入 %s (%d 条)", output_path, len(valid_items))

        report = self._build_report(valid_items, errors)
        if report_path:
            write_json(report, report_path)
            logger.info("[Aggregator] 评估报告已写入 %s", report_path)

        return report

    def _validate(self, item: OutputItem) -> list[str]:
        """基础格式校验。"""
        issues = []
        if not item.question.strip():
            issues.append("question 为空")
        if not item.answer.strip():
            issues.append("answer 为空")
        if len(item.answer) < 10:
            issues.append(f"answer 过短 ({len(item.answer)} 字符)")
        try:
            item.model_dump_json()
        except Exception as e:
            issues.append(f"JSON 序列化失败: {e}")
        return issues

    def _build_report(
        self,
        items: list[OutputItem],
        errors: list[dict],
    ) -> dict:
        """生成汇总报告。"""
        scored_items = [i for i in items if i.evaluation]
        scores = [i.evaluation.total_score for i in scored_items if i.evaluation]

        report = {
            "total_items": len(items),
            "items_with_errors": len(errors),
            "errors": errors,
        }

        if scores:
            passed = sum(1 for i in scored_items if i.evaluation and i.evaluation.passed)
            report.update({
                "scored_items": len(scored_items),
                "avg_score": round(sum(scores) / len(scores), 2),
                "max_score": max(scores),
                "min_score": min(scores),
                "pass_count": passed,
                "pass_rate": f"{passed / len(scored_items) * 100:.1f}%",
                "rework_stats": {
                    "total_reworks": sum(i.rework_count for i in items),
                    "items_reworked": sum(1 for i in items if i.rework_count > 0),
                },
            })

        return report
