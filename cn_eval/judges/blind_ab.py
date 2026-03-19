"""
Blind A/B 机制 — 随机分配 A/B 位置 + 位置交换消偏。

注意：UnifiedEvaluator 已内置多轮位置交换机制，本模块作为可选独立组件保留。
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from cn_eval.data_loader.schema import AlignedPair, PairwiseResult, DimensionScores, DIMENSIONS
from .base import BaseJudge
from .llm_judge import LLMJudge

logger = logging.getLogger(__name__)


class BlindABJudge:
    """Blind A/B 评审封装（独立组件）。"""

    def __init__(
        self,
        judge: BaseJudge,
        swap_ratio: float = 0.5,
        seed: int = 42,
    ):
        self.judge = judge
        self.swap_ratio = swap_ratio
        self._rng = random.Random(seed)

    async def evaluate_pair(self, pair: AlignedPair) -> PairwiseResult:
        do_swap = self._rng.random() < self.swap_ratio

        r1 = await self.judge.judge_single(
            prompt_text=pair.prompt_text,
            response_a=pair.baseline_response,
            response_b=pair.candidate_response,
        )

        result = PairwiseResult(
            prompt_id=pair.prompt_id,
            model_a=pair.baseline_version,
            model_b=pair.candidate_version,
            winner=r1.get("winner", "tie"),
            judge_id=self.judge.judge_id,
            scores_a=LLMJudge.parse_scores({"scores": r1.get("scores_a", {})}),
            scores_b=LLMJudge.parse_scores({"scores": r1.get("scores_b", {})}),
            reasoning=r1.get("reasoning", ""),
            position_swapped=False,
        )

        if not do_swap:
            return result

        r2 = await self.judge.judge_single(
            prompt_text=pair.prompt_text,
            response_a=pair.candidate_response,
            response_b=pair.baseline_response,
        )

        swapped_winner = r2.get("winner", "tie")
        swapped_winner_real = {"A": "B", "B": "A"}.get(swapped_winner, "tie")

        if result.winner == swapped_winner_real:
            result.metadata["consistency"] = "consistent"
            return result

        logger.warning(
            "[BlindAB] prompt %s 位置偏见: 原=%s, 换=%s → tie",
            pair.prompt_id, result.winner, swapped_winner_real,
        )
        result.winner = "tie"
        result.metadata["consistency"] = "position_bias_detected"
        result.position_swapped = True

        scores_a2 = LLMJudge.parse_scores({"scores": r2.get("scores_b", {})})
        scores_b2 = LLMJudge.parse_scores({"scores": r2.get("scores_a", {})})
        result.scores_a = self._avg_scores(result.scores_a, scores_a2)
        result.scores_b = self._avg_scores(result.scores_b, scores_b2)

        return result

    async def evaluate_batch(
        self,
        pairs: list[AlignedPair],
        concurrency: int = 5,
    ) -> list[PairwiseResult]:
        semaphore = asyncio.Semaphore(concurrency)
        results: list[PairwiseResult | None] = [None] * len(pairs)

        async def _run(idx: int, pair: AlignedPair) -> None:
            async with semaphore:
                try:
                    results[idx] = await self.evaluate_pair(pair)
                except Exception as e:
                    logger.error("[BlindAB] prompt %s 失败: %s", pair.prompt_id, e)
                    results[idx] = PairwiseResult(
                        prompt_id=pair.prompt_id,
                        model_a=pair.baseline_version,
                        model_b=pair.candidate_version,
                        metadata={"error": str(e)},
                    )

        await asyncio.gather(*[_run(i, p) for i, p in enumerate(pairs)])
        return [r for r in results if r is not None]

    @staticmethod
    def _avg_scores(s1: DimensionScores, s2: DimensionScores) -> DimensionScores:
        return DimensionScores(**{
            d: round((getattr(s1, d) + getattr(s2, d)) / 2, 1)
            for d in DIMENSIONS
        })
