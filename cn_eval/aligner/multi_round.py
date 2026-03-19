"""
多轮评测结果聚合 — 同一 (prompt_id, model_a, model_b) 的多次评测取聚合值。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from cn_eval.data_loader.schema import PairwiseResult, DimensionScores

logger = logging.getLogger(__name__)


class MultiRoundAggregator:
    """将多轮/多 Judge 的 PairwiseResult 按唯一键聚合。"""

    @staticmethod
    def aggregate(
        results: list[PairwiseResult],
        method: str = "majority_vote",
    ) -> list[PairwiseResult]:
        """
        按 (prompt_id, model_a, model_b) 聚合多条结果。

        method:
          - "majority_vote": 多数投票定 winner
          - "average": 分数取平均，总分高者胜
        """
        groups: dict[tuple, list[PairwiseResult]] = defaultdict(list)
        for r in results:
            key = (r.prompt_id, r.model_a, r.model_b)
            groups[key].append(r)

        aggregated: list[PairwiseResult] = []
        for key, group in groups.items():
            if len(group) == 1:
                aggregated.append(group[0])
                continue

            if method == "majority_vote":
                agg = MultiRoundAggregator._majority_vote(group)
            elif method == "average":
                agg = MultiRoundAggregator._average(group)
            else:
                raise ValueError(f"未知聚合方法: {method}")

            aggregated.append(agg)

        n_multi = sum(1 for g in groups.values() if len(g) > 1)
        logger.info(
            "[MultiRound] 聚合完成: %d 组 → %d 条 (%d 组有多轮, 方法=%s)",
            len(groups), len(aggregated), n_multi, method,
        )
        return aggregated

    @staticmethod
    def _majority_vote(group: list[PairwiseResult]) -> PairwiseResult:
        """多数投票。"""
        votes = {"A": 0, "B": 0, "tie": 0}
        for r in group:
            w = r.winner.upper() if r.winner else "tie"
            votes[w] = votes.get(w, 0) + 1

        winner = max(votes, key=lambda k: votes[k])
        if votes["A"] == votes["B"] and votes["A"] >= votes.get("tie", 0):
            winner = "tie"

        base = group[0].model_copy()
        base.winner = winner
        base.judge_id = f"majority({len(group)})"
        base.metadata["vote_counts"] = votes
        base.metadata["n_judges"] = len(group)
        return base

    @staticmethod
    def _average(group: list[PairwiseResult]) -> PairwiseResult:
        """分数取平均。"""
        n = len(group)

        def avg_scores(scores_list: list[DimensionScores]) -> DimensionScores:
            return DimensionScores(
                mode=sum(s.mode for s in scores_list) / n,
                structure=sum(s.structure for s in scores_list) / n,
                organization=sum(s.organization for s in scores_list) / n,
                fluency=sum(s.fluency for s in scores_list) / n,
                non_repetition=sum(s.non_repetition for s in scores_list) / n,
                task_fit=sum(s.task_fit for s in scores_list) / n,
            )

        avg_a = avg_scores([r.scores_a for r in group])
        avg_b = avg_scores([r.scores_b for r in group])

        if avg_a.mean() > avg_b.mean() + 0.1:
            winner = "A"
        elif avg_b.mean() > avg_a.mean() + 0.1:
            winner = "B"
        else:
            winner = "tie"

        base = group[0].model_copy()
        base.scores_a = avg_a
        base.scores_b = avg_b
        base.winner = winner
        base.judge_id = f"average({n})"
        base.metadata["n_judges"] = n
        return base
