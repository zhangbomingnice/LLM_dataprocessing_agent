"""
IF-Eval (Instruction Following) — 指令遵循硬验证。

检查模型回答是否满足 prompt 中的硬性约束：
长度、格式、语言、角色、内容包含/排除等。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from cn_eval.data_loader.schema import Prompt, ModelOutput, IFResult
from cn_eval.judges.llm_judge import LLMJudge
from cn_eval.utils.llm_client import LLMClient
from cn_eval.utils.config import EvalConfig
from cn_eval.utils.text import count_chars, split_paragraphs
from .base import BaseEvaluator

logger = logging.getLogger(__name__)


# 常见约束模式正则
_LENGTH_PATTERNS = [
    (r"(?:不超过|不多于|少于|最多)\s*(\d+)\s*(?:字|个字)", "max_chars"),
    (r"(?:不少于|至少|超过|最少)\s*(\d+)\s*(?:字|个字)", "min_chars"),
    (r"(?:约|大约|控制在)\s*(\d+)\s*(?:字|个字)", "approx_chars"),
    (r"(\d+)\s*(?:字|个字)\s*(?:以内|左右)", "max_chars"),
    (r"(\d+)\s*(?:段|个段落)", "paragraph_count"),
]

_FORMAT_PATTERNS = [
    (r"(?:请?用|以)\s*(?:列表|清单|要点)\s*(?:形式|格式)", "list"),
    (r"(?:请?用|以)\s*(?:表格)\s*(?:形式|格式)", "table"),
    (r"(?:请?用|以)\s*(?:编号|序号)", "numbered"),
    (r"(?:请?用|以)\s*(?:JSON|json)", "json"),
    (r"(?:请?用|以)\s*(?:markdown|Markdown)", "markdown"),
]

_LANGUAGE_PATTERNS = [
    (r"(?:用|请用|以)\s*(?:英文|英语)\s*(?:回答|作答|写)", "english"),
    (r"(?:用|请用|以)\s*(?:中文|汉语)\s*(?:回答|作答|写)", "chinese"),
    (r"(?:用|请用|以)\s*(?:日文|日语)\s*(?:回答|作答|写)", "japanese"),
]


class IFEvaluator(BaseEvaluator):
    """指令遵循评测器。"""

    name = "if_eval"

    def __init__(self, config: EvalConfig, client: LLMClient):
        super().__init__(config)
        self.client = client

    async def run(
        self,
        prompts: list[Prompt],
        model_outputs: list[ModelOutput],
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt_map = {p.prompt_id: p for p in prompts}
        results: list[IFResult] = []

        # 先用规则做确定性检查
        for output in model_outputs:
            prompt = prompt_map.get(output.prompt_id)
            if not prompt:
                continue

            constraints = self._extract_constraints(prompt.text)
            if not constraints:
                results.append(IFResult(
                    prompt_id=output.prompt_id,
                    model_version=output.model_version,
                    passed=True,
                    checks={},
                    details="无可检测约束",
                ))
                continue

            checks = self._rule_check(output.response, constraints)
            passed = all(v for v in checks.values())

            results.append(IFResult(
                prompt_id=output.prompt_id,
                model_version=output.model_version,
                passed=passed,
                checks=checks,
                details=f"约束检查: {constraints}",
            ))

        # 对规则不确定的条目，调用 LLM 辅助判断
        uncertain = [
            (i, r) for i, r in enumerate(results)
            if not r.checks and r.details == "无可检测约束"
        ]

        if uncertain and self.client:
            llm_judge = LLMJudge(
                client=self.client,
                prompt_template_path="configs/judge_prompts/if_eval_zh.txt",
                judge_id="llm_if_eval",
                temperature=0.0,
            )

            items = []
            for idx, result in uncertain:
                prompt = prompt_map.get(result.prompt_id)
                output = next(
                    (o for o in model_outputs if o.prompt_id == result.prompt_id), None,
                )
                if prompt and output:
                    items.append({
                        "prompt_id": result.prompt_id,
                        "prompt_text": prompt.text,
                        "response_a": output.response,
                        "index": idx,
                    })

            if items:
                llm_raw = await llm_judge.judge_batch(
                    items, concurrency=self.config.judge.concurrency,
                )
                for item, raw in zip(items, llm_raw):
                    idx = item["index"]
                    checks = raw.get("checks", {})
                    passed = raw.get("passed", True)
                    # 过滤 null 值
                    valid_checks = {k: v for k, v in checks.items() if v is not None}
                    if valid_checks:
                        passed = all(valid_checks.values())
                    results[idx] = IFResult(
                        prompt_id=item["prompt_id"],
                        model_version=results[idx].model_version,
                        passed=passed,
                        checks=valid_checks,
                        details=raw.get("details", "LLM 辅助检查"),
                    )

        summary = self._compute_summary(results)
        logger.info(
            "[IFEval] 完成 %d 条, 通过率 %.1f%%",
            len(results), summary.get("pass_rate", 0) * 100,
        )

        return {"results": results, "summary": summary}

    def _extract_constraints(self, prompt_text: str) -> dict[str, Any]:
        """从 prompt 中提取硬性约束。"""
        constraints: dict[str, Any] = {}

        for pattern, key in _LENGTH_PATTERNS:
            m = re.search(pattern, prompt_text)
            if m:
                constraints[key] = int(m.group(1))

        for pattern, fmt in _FORMAT_PATTERNS:
            if re.search(pattern, prompt_text):
                constraints["format"] = fmt

        for pattern, lang in _LANGUAGE_PATTERNS:
            if re.search(pattern, prompt_text):
                constraints["language"] = lang

        return constraints

    def _rule_check(self, response: str, constraints: dict) -> dict[str, bool]:
        """基于提取的约束做规则检查。"""
        checks: dict[str, bool] = {}
        char_count = count_chars(response)

        if "max_chars" in constraints:
            checks["length_max"] = char_count <= constraints["max_chars"] * 1.1
        if "min_chars" in constraints:
            checks["length_min"] = char_count >= constraints["min_chars"] * 0.9
        if "approx_chars" in constraints:
            target = constraints["approx_chars"]
            checks["length_approx"] = target * 0.7 <= char_count <= target * 1.3
        if "paragraph_count" in constraints:
            paras = split_paragraphs(response)
            target = constraints["paragraph_count"]
            checks["paragraph_count"] = abs(len(paras) - target) <= 1

        fmt = constraints.get("format")
        if fmt == "list":
            checks["format_list"] = bool(re.search(r'^\s*[-*•]\s', response, re.MULTILINE))
        elif fmt == "numbered":
            checks["format_numbered"] = bool(re.search(r'^\s*\d+[.、)]\s', response, re.MULTILINE))
        elif fmt == "table":
            checks["format_table"] = "|" in response and "---" in response
        elif fmt == "json":
            checks["format_json"] = response.strip().startswith("{") or response.strip().startswith("[")

        lang = constraints.get("language")
        if lang == "english":
            cn_ratio = len(re.findall(r'[\u4e00-\u9fff]', response)) / max(len(response), 1)
            checks["language_english"] = cn_ratio < 0.1
        elif lang == "chinese":
            cn_ratio = len(re.findall(r'[\u4e00-\u9fff]', response)) / max(len(response), 1)
            checks["language_chinese"] = cn_ratio > 0.3

        return checks

    def _compute_summary(self, results: list[IFResult]) -> dict[str, Any]:
        if not results:
            return {}

        n = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = n - passed

        check_stats: dict[str, dict] = {}
        for r in results:
            for check_name, check_val in r.checks.items():
                if check_name not in check_stats:
                    check_stats[check_name] = {"total": 0, "passed": 0}
                check_stats[check_name]["total"] += 1
                if check_val:
                    check_stats[check_name]["passed"] += 1

        return {
            "total": n,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / n,
            "check_stats": check_stats,
        }
