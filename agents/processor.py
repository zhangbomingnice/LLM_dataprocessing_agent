from __future__ import annotations

import asyncio
import logging
from typing import Any

from config import LLMConfig
from utils.schema import CorpusItem, PlannerOutput

from .base import BaseAgent

logger = logging.getLogger(__name__)

PROCESSOR_FALLBACK_PROMPT = """\
你是一名专业的语料生成专家。请根据给定的问题，生成一个高质量、结构清晰、逻辑严谨的回答。
回答应当：
- 准确、完备地覆盖问题要点
- 使用清晰的分步结构
- 语言流畅自然
"""


class ProcessorAgent(BaseAgent):
    """Agent 2: 根据 Planner 生成的动态 Prompt，逐条生成/重写答案。"""

    name = "Processor"

    def __init__(self, llm_config: LLMConfig, plan: PlannerOutput | None = None):
        system_prompt = (
            plan.processor_system_prompt if plan and plan.processor_system_prompt
            else PROCESSOR_FALLBACK_PROMPT
        )
        if plan and plan.few_shot_example:
            system_prompt += f"\n\n# Few-shot 示例\n{plan.few_shot_example}"
        if plan and plan.gold_standard:
            system_prompt += f"\n\n# 质量标准\n{plan.gold_standard}"

        super().__init__(llm_config, system_prompt=system_prompt)
        self.plan = plan

    async def run(self, item: CorpusItem) -> str:
        """为单条语料生成答案。"""
        user_msg = f"请为以下问题生成高质量回答：\n\n## 问题\n{item.question}"
        if item.answer:
            user_msg += f"\n\n## 原始回答（供参考，请改进或重写）\n{item.answer}"

        logger.debug("[Processor] 处理条目 %s", item.id)
        return await self._call_llm(self._build_messages(user_msg))

    async def run_batch(
        self,
        items: list[CorpusItem],
        concurrency: int = 5,
    ) -> dict[str | int, str]:
        """并发批量处理，返回 {id: answer} 映射。"""
        semaphore = asyncio.Semaphore(concurrency)
        results: dict[str | int, str] = {}

        async def _process(item: CorpusItem) -> None:
            async with semaphore:
                answer = await self.run(item)
                results[item.id] = answer

        tasks = [_process(item) for item in items]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("[Processor] 批量处理完成: %d / %d", len(results), len(items))
        return results

    async def rework(self, item: CorpusItem, feedback: str) -> str:
        """根据 Evaluator 的反馈重写答案。"""
        user_msg = (
            f"请根据以下反馈，重写回答：\n\n"
            f"## 问题\n{item.question}\n\n"
            f"## 当前回答\n{item.answer}\n\n"
            f"## 评审反馈\n{feedback}\n\n"
            f"请输出改进后的完整回答。"
        )
        logger.debug("[Processor] 重写条目 %s", item.id)
        return await self._call_llm(self._build_messages(user_msg))
