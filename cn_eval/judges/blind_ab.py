"""
Blind A/B 机制 — 随机分配 A/B 位置 + 位置交换消偏。
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from cn_eval.data_loader.schema import AlignedPair, PairwiseResult, DimensionScores
from .base import BaseJudge
from .llm_judge import LLMJudge

logger = logging.getLogger(__name__)


class BlindABJudge:
    """
    Blind A/B 评审封装。

    对每条数据做两次评审（原序 + 交换位置），
    如果结论不一致，按 swap_ratio 采样做交换评审。
    """

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
        """对单个对齐对做 Blind A/B 评审。"""
        do_swap = self._rng.random() < self.swap_ratio

        # 第一次评审：原始顺序
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

        # 第二次评审：交换 A/B 位置
        r2 = await self.judge.judge_single(
            prompt_text=pair.prompt_text,
            response_a=pair.candidate_response,
            response_b=pair.baseline_response,
        )

        swapped_winner = r2.get("winner", "tie")
        # 将交换后的结果还原
        if swapped_winner == "A":
            swapped_winner_real = "B"
        elif swapped_winner == "B":
            swapped_winner_real = "A"
        else:
            swapped_winner_real = "tie"

        # 两次评审一致 → 可信
        if result.winner == swapped_winner_real:
            result.metadata["consistency"] = "consistent"
            return result

        # 不一致 → 标记并取保守结果（倾向 tie）
        logger.warning(
            "[BlindAB] prompt %s 位置偏见检出: 原序=%s, 交换后=%s → 判为 tie",
            pair.prompt_id, result.winner, swapped_winner_real,
        )
        result.winner = "tie"
        result.metadata["consistency"] = "position_bias_detected"
        result.metadata["original_winner"] = r1.get("winner", "")
        result.metadata["swapped_winner"] = swapped_winner
        result.position_swapped = True

        # 取两次分数的平均
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
        """批量 Blind A/B 评审。"""
        semaphore = asyncio.Semaphore(concurrency)
        results: list[PairwiseResult | None] = [None] * len(pairs)

        async def _run(idx: int, pair: AlignedPair) -> None:
            async with semaphore:
                try:
                    results[idx] = await self.evaluate_pair(pair)
                except Exception as e:
                    logger.error("[BlindAB] prompt %s 评审失败: %s", pair.prompt_id, e)
                    results[idx] = PairwiseResult(
                        prompt_id=pair.prompt_id,
                        model_a=pair.baseline_version,
                        model_b=pair.candidate_version,
                        metadata={"error": str(e)},
                    )

        await asyncio.gather(*[_run(i, p) for i, p in enumerate(pairs)])

        n_bias = sum(1 for r in results if r and r.metadata.get("consistency") == "position_bias_detected")
        logger.info(
            "[BlindAB] 完成 %d 条评审 (位置偏见检出 %d 条, %.1f%%)",
            len(pairs), n_bias, n_bias / max(len(pairs), 1) * 100,
        )
        return [r for r in results if r is not None]

    @staticmethod
    def _avg_scores(s1: DimensionScores, s2: DimensionScores) -> DimensionScores:
        return DimensionScores(
            mode=(s1.mode + s2.mode) / 2,
            structure=(s1.structure + s2.structure) / 2,
            organization=(s1.organization + s2.organization) / 2,
            fluency=(s1.fluency + s2.fluency) / 2,
            non_repetition=(s1.non_repetition + s2.non_repetition) / 2,
            task_fit=(s1.task_fit + s2.task_fit) / 2,
        )
