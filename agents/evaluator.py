from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from config import LLMConfig
from utils.schema import CorpusItem, PlannerOutput, EvalResult, DimensionScore

from .base import BaseAgent

logger = logging.getLogger(__name__)

EVALUATOR_FALLBACK_PROMPT = """\
你是一名严格的语料质量评审员。请从以下四个维度对回答进行逐一评分（0-10 分）：

1. **准确性 (Accuracy)**：事实是否正确？推导是否无误？
2. **完备性 (Completeness)**：是否完整覆盖问题所有要点？
3. **逻辑严谨性 (Logic)**：论证链条是否连贯？有无逻辑跳跃？
4. **格式依从性 (Format)**：是否符合格式要求（语言、结构、排版）？

# 输出要求
严格输出以下 JSON（不要包含 ```json 标记）：
{
  "total_score": 综合评分 (0-10，取四维度加权平均，准确性权重 0.35，完备性 0.25，逻辑 0.25，格式 0.15),
  "dimensions": [
    {"dimension": "准确性", "score": 0-10, "reason": "评分理由"},
    {"dimension": "完备性", "score": 0-10, "reason": "评分理由"},
    {"dimension": "逻辑严谨性", "score": 0-10, "reason": "评分理由"},
    {"dimension": "格式依从性", "score": 0-10, "reason": "评分理由"}
  ],
  "suggestion": "改进建议（如果总分 >= 7 可写'无'）"
}
"""

DIMENSION_WEIGHTS = {
    "准确性": 0.35,
    "完备性": 0.25,
    "逻辑严谨性": 0.25,
    "格式依从性": 0.15,
}


class EvaluatorAgent(BaseAgent):
    """Agent 3: 对答案进行多维度评分，不合格则给出改进建议。"""

    name = "Evaluator"

    def __init__(
        self,
        llm_config: LLMConfig,
        plan: PlannerOutput | None = None,
        pass_threshold: float = 7.0,
    ):
        system_prompt = (
            plan.evaluator_system_prompt if plan and plan.evaluator_system_prompt
            else EVALUATOR_FALLBACK_PROMPT
        )
        if plan and plan.gold_standard:
            system_prompt += f"\n\n# 质量标准参考\n{plan.gold_standard}"

        super().__init__(llm_config, system_prompt=system_prompt)
        self.pass_threshold = pass_threshold

    async def run(self, item_id: str | int, question: str, answer: str) -> EvalResult:
        """对单条 QA 进行评分（带双重验证）。"""
        user_msg = f"请评估以下问答对的答案质量：\n\n## 问题\n{question}\n\n## 回答\n{answer}"

        logger.debug("[Evaluator] 评估条目 %s", item_id)

        # 第一轮评分
        result1 = await self._call_llm_json(
            self._build_messages(user_msg),
            temperature=self.llm_config.eval_temperature,
        )
        eval1 = self._parse_eval_result(item_id, result1)

        # 第二轮验证：用不同温度再评一次，检查一致性
        result2 = await self._call_llm_json(
            self._build_messages(user_msg),
            temperature=min(self.llm_config.eval_temperature + 0.15, 0.6),
        )
        eval2 = self._parse_eval_result(item_id, result2)

        score_diff = abs(eval1.total_score - eval2.total_score)
        if score_diff > 2.0:
            # 分差过大，取第三次评分做仲裁
            logger.info(
                "[Evaluator] 条目 %s 两轮评分差异 %.1f，启动仲裁",
                item_id, score_diff,
            )
            arbitration_msg = (
                f"以下是两位评审员对同一问答的评分，请你作为仲裁员给出最终评分：\n\n"
                f"## 问题\n{question}\n\n## 回答\n{answer}\n\n"
                f"## 评审 A 评分: {eval1.total_score:.1f}\n"
                f"## 评审 B 评分: {eval2.total_score:.1f}\n\n"
                f"请综合两方意见，输出你的最终评分。"
            )
            result3 = await self._call_llm_json(
                self._build_messages(arbitration_msg),
                temperature=self.llm_config.eval_temperature,
            )
            return self._parse_eval_result(item_id, result3)

        # 分差在可接受范围内，取平均
        return self._merge_evals(item_id, eval1, eval2)

    def _parse_eval_result(self, item_id: str | int, result: dict) -> EvalResult:
        """将 LLM 返回的 JSON 解析为 EvalResult。"""
        dimensions = []
        for d in result.get("dimensions", []):
            dimensions.append(DimensionScore(
                dimension=d.get("dimension", "未知"),
                score=float(d.get("score", 0)),
                reason=d.get("reason", ""),
            ))

        if dimensions:
            total = sum(
                d.score * DIMENSION_WEIGHTS.get(d.dimension, 0.25)
                for d in dimensions
            )
        else:
            total = float(result.get("total_score", 0))

        return EvalResult(
            item_id=item_id,
            total_score=round(total, 2),
            passed=total >= self.pass_threshold,
            dimensions=dimensions,
            suggestion=result.get("suggestion", ""),
        )

    @staticmethod
    def _merge_evals(item_id: str | int, a: EvalResult, b: EvalResult) -> EvalResult:
        """两轮评分取平均，合并维度说明。"""
        avg_score = round((a.total_score + b.total_score) / 2, 2)
        merged_dims = []
        for da, db in zip(a.dimensions, b.dimensions):
            merged_dims.append(DimensionScore(
                dimension=da.dimension,
                score=round((da.score + db.score) / 2, 1),
                reason=da.reason if da.reason else db.reason,
            ))

        suggestion = a.suggestion if a.suggestion and a.suggestion != "无" else b.suggestion
        return EvalResult(
            item_id=item_id,
            total_score=avg_score,
            passed=a.passed and b.passed,
            dimensions=merged_dims if merged_dims else a.dimensions,
            suggestion=suggestion,
        )

    async def run_batch(
        self,
        items: list[tuple[str | int, str, str]],
        concurrency: int = 5,
    ) -> list[EvalResult]:
        """并发批量评估。items 为 (id, question, answer) 元组列表。"""
        semaphore = asyncio.Semaphore(concurrency)
        results: list[EvalResult] = []
        lock = asyncio.Lock()

        async def _evaluate(item_id: str | int, question: str, answer: str) -> None:
            async with semaphore:
                res = await self.run(item_id, question, answer)
                async with lock:
                    results.append(res)

        tasks = [_evaluate(iid, q, a) for iid, q, a in items]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("[Evaluator] 批量评估完成: %d / %d", len(results), len(items))
        return sorted(results, key=lambda r: str(r.item_id))
