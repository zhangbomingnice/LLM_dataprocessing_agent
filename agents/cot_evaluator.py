from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from config import LLMConfig
from utils.schema import (
    PlannerOutput, CoTStep, CoTEvalResult, StepVerification,
)

from .base import BaseAgent

logger = logging.getLogger(__name__)

COT_EVAL_SYSTEM_PROMPT = """\
你是一名严格的数理推导审查专家。你的任务是对给定的分步推理过程进行**逐步骤验证**。

# 审查要求
对每一个推理步骤，你需要判断：
1. 该步骤本身是否正确（计算、公式、逻辑）
2. 该步骤与上一步是否衔接（是否有逻辑跳跃）
3. 该步骤使用的公式/定理是否适用于当前情境

# 错误类型分类
- 无误：该步骤完全正确
- 计算错误：数值计算有误
- 逻辑跳跃：缺少中间推导步骤
- 公式误用：使用了不适用的公式或定理
- 概念混淆：混淆了相关但不同的概念
- 条件遗漏：忽略了题目给出的约束条件
- 符号错误：变量符号使用不一致或错误
- 单位错误：物理量单位换算或使用有误

# 输出格式
严格输出以下 JSON（不要包含 ```json 标记）：
{
  "step_verifications": [
    {
      "step_number": 1,
      "is_correct": true/false,
      "error_type": "无误/计算错误/逻辑跳跃/...",
      "explanation": "该步骤正确/错误的具体原因",
      "suggested_fix": "如有错误，给出修正建议（正确则留空）"
    }
  ],
  "chain_coherence": 0-10 的评分（推理链整体连贯性）,
  "final_answer_correct": true/false,
  "overall_score": 0-10 的综合评分,
  "suggestion": "整体改进建议"
}
"""


class CoTEvaluatorAgent(BaseAgent):
    """CoT 专用评估器：逐步骤验证推理链的正确性。"""

    name = "CoT-Evaluator"

    def __init__(
        self,
        llm_config: LLMConfig,
        plan: PlannerOutput | None = None,
        pass_threshold: float = 7.0,
    ):
        system_prompt = COT_EVAL_SYSTEM_PROMPT
        if plan and plan.evaluator_system_prompt:
            system_prompt += f"\n\n# 补充评审要求\n{plan.evaluator_system_prompt}"
        if plan and plan.gold_standard:
            system_prompt += f"\n\n# 质量标准参考\n{plan.gold_standard}"

        super().__init__(llm_config, system_prompt=system_prompt)
        self.pass_threshold = pass_threshold

    async def run(
        self,
        item_id: str | int,
        question: str,
        steps: list[CoTStep],
        final_answer: str,
        reference_answer: str = "",
    ) -> CoTEvalResult:
        """对单条 CoT 进行逐步骤验证。"""
        steps_text = "\n".join(
            f"Step {s.step_number} [{s.step_type}]: {s.content}"
            + (f"\n  公式: {s.formula}" if s.formula else "")
            for s in steps
        )

        user_msg = (
            f"请逐步骤验证以下推理过程：\n\n"
            f"## 问题\n{question}\n\n"
            f"## 推理步骤\n{steps_text}\n\n"
            f"## 最终答案\n{final_answer}"
        )
        if reference_answer:
            user_msg += f"\n\n## 参考答案\n{reference_answer}"

        logger.debug("[CoT-Evaluator] 验证条目 %s (%d 步)", item_id, len(steps))

        result = await self._call_llm_json(
            self._build_messages(user_msg),
            temperature=self.llm_config.eval_temperature,
        )

        verifications = []
        for v in result.get("step_verifications", []):
            verifications.append(StepVerification(
                step_number=v.get("step_number", 0),
                is_correct=v.get("is_correct", True),
                error_type=v.get("error_type", "无误"),
                explanation=v.get("explanation", ""),
                suggested_fix=v.get("suggested_fix", ""),
            ))

        total_steps = len(steps)
        correct_steps = sum(1 for v in verifications if v.is_correct)
        first_error = next(
            (v.step_number for v in verifications if not v.is_correct),
            None,
        )
        step_accuracy = correct_steps / total_steps if total_steps > 0 else 0

        overall_score = float(result.get("overall_score", step_accuracy * 10))
        chain_coherence = float(result.get("chain_coherence", 0))
        final_correct = result.get("final_answer_correct", False)

        passed = overall_score >= self.pass_threshold and final_correct

        return CoTEvalResult(
            item_id=item_id,
            total_steps=total_steps,
            correct_steps=correct_steps,
            first_error_step=first_error,
            step_accuracy=round(step_accuracy, 4),
            overall_score=round(overall_score, 2),
            passed=passed,
            step_verifications=verifications,
            chain_coherence=chain_coherence,
            final_answer_correct=final_correct,
            suggestion=result.get("suggestion", ""),
        )

    async def run_batch(
        self,
        items: list[tuple[str | int, str, list[CoTStep], str, str]],
        concurrency: int = 5,
    ) -> list[CoTEvalResult]:
        """并发批量验证。items: (id, question, steps, final_answer, ref_answer)。"""
        semaphore = asyncio.Semaphore(concurrency)
        results: list[CoTEvalResult] = []
        lock = asyncio.Lock()

        async def _eval(
            item_id: str | int,
            question: str,
            steps: list[CoTStep],
            final_answer: str,
            ref_answer: str,
        ) -> None:
            async with semaphore:
                try:
                    res = await self.run(item_id, question, steps, final_answer, ref_answer)
                    async with lock:
                        results.append(res)
                except Exception as e:
                    logger.error("[CoT-Evaluator] 条目 %s 验证失败: %s", item_id, e)

        tasks = [_eval(*item) for item in items]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("[CoT-Evaluator] 批量验证完成: %d / %d", len(results), len(items))
        return sorted(results, key=lambda r: str(r.item_id))

    def format_feedback(self, eval_result: CoTEvalResult) -> str:
        """将评估结果格式化为可供 Processor 使用的反馈文本。"""
        lines = []
        for v in eval_result.step_verifications:
            if not v.is_correct:
                lines.append(
                    f"Step {v.step_number} 错误 [{v.error_type}]: {v.explanation}"
                )
                if v.suggested_fix:
                    lines.append(f"  建议修正: {v.suggested_fix}")

        if not eval_result.final_answer_correct:
            lines.append("最终答案不正确，需要修正。")

        lines.append(f"推理链连贯性评分: {eval_result.chain_coherence}/10")

        if eval_result.suggestion:
            lines.append(f"总体建议: {eval_result.suggestion}")

        return "\n".join(lines)
