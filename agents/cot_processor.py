from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from config import LLMConfig
from utils.schema import CorpusItem, PlannerOutput, CoTStep

from .base import BaseAgent

logger = logging.getLogger(__name__)

COT_SYSTEM_PROMPT = """\
你是一名专业的数理推导教师，擅长将复杂问题拆解为严格的分步推理过程。

# 核心任务
为给定的问题生成高质量的 Chain-of-Thought（思维链）推理数据。每一步推导必须：
1. 有明确的步骤编号和步骤类型标签
2. 逻辑严谨，步步有据，不跳步
3. 涉及公式时使用 LaTeX 格式
4. 最后给出明确的结论

# 步骤类型标签（根据实际情况选用）
- 审题：分析题目条件和求解目标
- 建模：建立数学/物理模型
- 公式推导：推导或引用相关公式
- 代入计算：将具体数值代入计算
- 变换化简：数学变换和化简
- 逻辑推理：逻辑分析和推断
- 分类讨论：对不同情况分别讨论
- 验证：验证结果的正确性
- 结论：总结最终答案

# 输出格式
严格输出以下 JSON（不要包含 ```json 标记）：
{
  "thinking": "完整的推理过程文本（供 <think> 标签使用）",
  "steps": [
    {
      "step_number": 1,
      "step_type": "审题",
      "content": "该步骤的详细推理内容",
      "formula": "该步涉及的核心公式（LaTeX，没有则留空）"
    }
  ],
  "final_answer": "最终答案的简洁表述"
}
"""


class CoTProcessorAgent(BaseAgent):
    """CoT 专用处理器：生成带步骤标签的思维链推理数据。"""

    name = "CoT-Processor"

    def __init__(self, llm_config: LLMConfig, plan: PlannerOutput | None = None):
        system_prompt = COT_SYSTEM_PROMPT
        if plan and plan.processor_system_prompt:
            system_prompt += f"\n\n# 补充要求\n{plan.processor_system_prompt}"
        if plan and plan.few_shot_example:
            system_prompt += f"\n\n# Few-shot 示例\n{plan.few_shot_example}"
        if plan and plan.gold_standard:
            system_prompt += f"\n\n# 质量标准\n{plan.gold_standard}"

        super().__init__(llm_config, system_prompt=system_prompt)
        self.plan = plan

    async def run(self, item: CorpusItem) -> dict:
        """为单条问题生成 CoT 推理链。返回 {thinking, steps, final_answer}。"""
        user_msg = f"请为以下问题生成完整的分步推理过程：\n\n## 问题\n{item.question}"
        if item.answer:
            user_msg += f"\n\n## 参考答案（最终结果应与此一致或更正它）\n{item.answer}"

        logger.debug("[CoT-Processor] 处理条目 %s", item.id)
        result = await self._call_llm_json(self._build_messages(user_msg))

        steps = []
        for s in result.get("steps", []):
            steps.append(CoTStep(
                step_number=s.get("step_number", 0),
                step_type=s.get("step_type", ""),
                content=s.get("content", ""),
                formula=s.get("formula", ""),
            ))

        return {
            "thinking": result.get("thinking", ""),
            "steps": steps,
            "final_answer": result.get("final_answer", ""),
        }

    async def run_batch(
        self,
        items: list[CorpusItem],
        concurrency: int = 5,
    ) -> dict[str | int, dict]:
        """并发批量处理。"""
        semaphore = asyncio.Semaphore(concurrency)
        results: dict[str | int, dict] = {}

        async def _process(item: CorpusItem) -> None:
            async with semaphore:
                try:
                    result = await self.run(item)
                    results[item.id] = result
                except Exception as e:
                    logger.error("[CoT-Processor] 条目 %s 处理失败: %s", item.id, e)

        await asyncio.gather(*[_process(item) for item in items], return_exceptions=True)
        logger.info("[CoT-Processor] 批量处理完成: %d / %d", len(results), len(items))
        return results

    async def rework(self, item: CorpusItem, feedback: str, current_steps: list[CoTStep]) -> dict:
        """根据逐步验证的反馈重写 CoT。"""
        steps_text = "\n".join(
            f"Step {s.step_number} [{s.step_type}]: {s.content}"
            for s in current_steps
        )
        user_msg = (
            f"以下推理过程存在问题，请根据反馈修正并重新生成完整的分步推理：\n\n"
            f"## 问题\n{item.question}\n\n"
            f"## 当前推理步骤\n{steps_text}\n\n"
            f"## 逐步验证反馈\n{feedback}\n\n"
            f"请修正错误步骤，输出完整的新推理过程。"
        )

        result = await self._call_llm_json(self._build_messages(user_msg))
        steps = [
            CoTStep(
                step_number=s.get("step_number", 0),
                step_type=s.get("step_type", ""),
                content=s.get("content", ""),
                formula=s.get("formula", ""),
            )
            for s in result.get("steps", [])
        ]

        return {
            "thinking": result.get("thinking", ""),
            "steps": steps,
            "final_answer": result.get("final_answer", ""),
        }


def format_cot_answer(thinking: str, steps: list[CoTStep], final_answer: str) -> str:
    """将 CoT 结果格式化为 SFT 训练用的标准答案格式。"""
    parts = []

    parts.append("<think>")
    parts.append(thinking)
    parts.append("</think>")
    parts.append("")

    for s in steps:
        label = f"**Step {s.step_number}" + (f" [{s.step_type}]" if s.step_type else "") + "**"
        parts.append(label)
        parts.append(s.content)
        if s.formula:
            parts.append(f"$$\n{s.formula}\n$$")
        parts.append("")

    parts.append(f"**最终答案：** {final_answer}")
    return "\n".join(parts)
