"""
数据增强模块 — 从一道题生成多个变体。

增强策略：
1. 数值替换：改变题目中的具体数字
2. 条件变换：增加/修改约束条件
3. 逆向出题：给答案反推问题
4. 同类衍生：同知识点不同题型
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from config import LLMConfig
from agents.base import BaseAgent
from utils.schema import CorpusItem

logger = logging.getLogger(__name__)

AUGMENT_SYSTEM_PROMPT = """\
你是一名专业的题目变体生成专家。给定一道原始题目，请生成指定数量的变体题目。

# 变体策略（请混合使用）
1. **数值替换**：改变题目中的具体数字（如半径 5→8，高度 20→30）
2. **条件变换**：增加、删除或修改约束条件
3. **同类衍生**：保持同一知识点，换一种题型或问法
4. **难度调整**：生成更简单或更难的版本

# 要求
- 每个变体必须是一道完整的、可独立解答的新题目
- 变体之间应有足够的差异性
- 保持与原题相同的知识点领域
- 如果原题有答案，也请给出变体题的参考答案

# 输出格式
严格输出 JSON（不要包含 ```json 标记）：
{
  "variants": [
    {"question": "变体题目1", "answer": "参考答案1", "strategy": "使用的变体策略"},
    {"question": "变体题目2", "answer": "参考答案2", "strategy": "使用的变体策略"}
  ]
}
"""


class DataAugmentor(BaseAgent):
    """数据增强器：从一道题生成多个变体。"""

    name = "Augmentor"

    def __init__(self, llm_config: LLMConfig, n_variants: int = 3):
        super().__init__(llm_config, system_prompt=AUGMENT_SYSTEM_PROMPT)
        self.n_variants = n_variants

    async def run(self, item: CorpusItem) -> list[CorpusItem]:
        """为单条数据生成变体。"""
        user_msg = (
            f"请为以下题目生成 {self.n_variants} 个变体：\n\n"
            f"## 原始题目\n{item.question}"
        )
        if item.answer:
            user_msg += f"\n\n## 原始答案\n{item.answer}"

        logger.debug("[Augmentor] 增强条目 %s", item.id)
        result = await self._call_llm_json(self._build_messages(user_msg))

        variants = []
        for idx, v in enumerate(result.get("variants", [])):
            variants.append(CorpusItem(
                id=f"{item.id}_aug_{idx + 1}",
                question=v.get("question", ""),
                answer=v.get("answer", ""),
                metadata={
                    "source_id": str(item.id),
                    "augment_strategy": v.get("strategy", ""),
                },
            ))

        logger.debug("[Augmentor] 条目 %s 生成 %d 个变体", item.id, len(variants))
        return variants

    async def run_batch(
        self,
        items: list[CorpusItem],
        concurrency: int = 5,
    ) -> list[CorpusItem]:
        """并发批量增强。返回所有生成的变体。"""
        semaphore = asyncio.Semaphore(concurrency)
        all_variants: list[CorpusItem] = []
        lock = asyncio.Lock()

        async def _augment(item: CorpusItem) -> None:
            async with semaphore:
                try:
                    variants = await self.run(item)
                    async with lock:
                        all_variants.extend(variants)
                except Exception as e:
                    logger.error("[Augmentor] 条目 %s 增强失败: %s", item.id, e)

        await asyncio.gather(*[_augment(item) for item in items], return_exceptions=True)

        logger.info(
            "[Augmentor] 批量增强完成: %d 条原始 → %d 条变体",
            len(items), len(all_variants),
        )
        return all_variants
