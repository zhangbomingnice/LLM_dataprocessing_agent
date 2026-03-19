"""
LLM Judge — 使用大模型作为评审员。

支持自定义 prompt 模板、温度控制、JSON 解析。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from cn_eval.data_loader.schema import DimensionScores, DIMENSIONS
from cn_eval.utils.llm_client import LLMClient
from .base import BaseJudge

logger = logging.getLogger(__name__)


class LLMJudge(BaseJudge):
    """基于 LLM 的评审员。"""

    def __init__(
        self,
        client: LLMClient,
        system_prompt: str = "",
        prompt_template_path: str = "",
        judge_id: str = "llm_primary",
        temperature: float = 0.0,
    ):
        self.client = client
        self.judge_id = judge_id
        self.temperature = temperature

        if prompt_template_path and Path(prompt_template_path).exists():
            self._system_prompt = Path(prompt_template_path).read_text(encoding="utf-8")
        elif system_prompt:
            self._system_prompt = system_prompt
        else:
            self._system_prompt = self._default_system_prompt()

    async def judge_single(
        self,
        prompt_text: str,
        response_a: str,
        response_b: str = "",
        **kwargs: Any,
    ) -> dict:
        if response_b:
            user_msg = self._build_pairwise_prompt(prompt_text, response_a, response_b)
        else:
            user_msg = self._build_single_prompt(prompt_text, response_a)

        result = await self.client.judge(
            system_prompt=self._system_prompt,
            user_prompt=user_msg,
            temperature=self.temperature,
        )

        result["judge_id"] = self.judge_id
        return result

    async def judge_batch(
        self,
        items: list[dict],
        concurrency: int = 5,
        **kwargs: Any,
    ) -> list[dict]:
        semaphore = asyncio.Semaphore(concurrency)
        results: list[dict | None] = [None] * len(items)

        async def _run(idx: int, item: dict) -> None:
            async with semaphore:
                try:
                    r = await self.judge_single(
                        prompt_text=item.get("prompt_text", ""),
                        response_a=item.get("response_a", ""),
                        response_b=item.get("response_b", ""),
                    )
                    r["prompt_id"] = item.get("prompt_id", "")
                    results[idx] = r
                except Exception as e:
                    logger.error("[LLMJudge] 第 %d 条评审失败: %s", idx, e)
                    results[idx] = {
                        "prompt_id": item.get("prompt_id", ""),
                        "error": str(e),
                        "judge_id": self.judge_id,
                    }

        await asyncio.gather(*[_run(i, item) for i, item in enumerate(items)])
        logger.info("[LLMJudge:%s] 批量评审完成: %d 条", self.judge_id, len(items))
        return [r or {"error": "unknown"} for r in results]

    def _build_pairwise_prompt(self, prompt: str, resp_a: str, resp_b: str) -> str:
        return (
            f"## 问题\n{prompt}\n\n"
            f"## 回答 A\n{resp_a}\n\n"
            f"## 回答 B\n{resp_b}"
        )

    def _build_single_prompt(self, prompt: str, response: str) -> str:
        return (
            f"## 问题\n{prompt}\n\n"
            f"## 模型回答\n{response}"
        )

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "你是一位专业的中文文本质量评审专家。请根据多个维度对回答进行评分，"
            "以 JSON 格式输出结果。每个维度分数为 1-5 的整数。"
        )

    @staticmethod
    def parse_scores(raw: dict) -> DimensionScores:
        """将 LLM 返回的 dict 解析为 DimensionScores。"""
        scores = raw.get("scores", raw.get("scores_a", raw))
        return DimensionScores(**{
            dim: float(scores.get(dim, 0)) for dim in DIMENSIONS
        })
