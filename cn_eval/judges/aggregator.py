"""
Judge 聚合器 — 多 Judge 结果的融合与仲裁。
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from cn_eval.data_loader.schema import PairwiseResult, DimensionScores

logger = logging.getLogger(__name__)


class JudgeAggregator:
    """
    多 Judge 结果聚合器。

    策略:
      - majority_vote: 多数投票
      - weighted_vote: 加权投票（给主 Judge 更高权重）
      - conservative: 不一致时判 tie
    """

    def __init__(
        self,
        strategy: str = "majority_vote",
        primary_judge_id: str = "llm_primary",
        primary_weight: float = 2.0,
    ):
        self.strategy = strategy
        self.primary_judge_id = primary_judge_id
        self.primary_weight = primary_weight

    def aggregate(
        self,
        results_by_judge: dict[str, list[PairwiseResult]],
    ) -> list[PairwiseResult]:
        """
        将多个 Judge 的结果聚合为一份最终结果。

        results_by_judge: {judge_id: [PairwiseResult, ...]}
        """
        # 按 prompt_id 分组
        grouped: dict[str, dict[str, PairwiseResult]] = {}
        for jid, results in results_by_judge.items():
            for r in results:
                grouped.setdefault(r.prompt_id, {})[jid] = r

        aggregated = []
        for pid, judge_results in grouped.items():
            agg = self._aggregate_one(pid, judge_results)
            aggregated.append(agg)

        n_tie = sum(1 for r in aggregated if r.winner == "tie")
        logger.info(
            "[Aggregator] 聚合完成: %d 条 (%d judges, tie 率 %.1f%%, 策略=%s)",
            len(aggregated), len(results_by_judge), n_tie / max(len(aggregated), 1) * 100,
            self.strategy,
        )
        return aggregated

    def _aggregate_one(
        self,
        prompt_id: str,
        judge_results: dict[str, PairwiseResult],
    ) -> PairwiseResult:
        if len(judge_results) == 1:
            return list(judge_results.values())[0]

        if self.strategy == "majority_vote":
            return self._majority_vote(prompt_id, judge_results)
        elif self.strategy == "weighted_vote":
            return self._weighted_vote(prompt_id, judge_results)
        elif self.strategy == "conservative":
            return self._conservative(prompt_id, judge_results)
        else:
            raise ValueError(f"未知聚合策略: {self.strategy}")

    def _majority_vote(self, pid: str, jr: dict[str, PairwiseResult]) -> PairwiseResult:
        votes = Counter(r.winner for r in jr.values())
        winner = votes.most_common(1)[0][0]

        if votes.most_common(1)[0][1] == 1 and len(jr) > 2:
            winner = "tie"

        base = list(jr.values())[0].model_copy()
        base.winner = winner
        base.judge_id = f"agg_majority({len(jr)})"
        base.metadata["judge_votes"] = dict(votes)
        base.metadata["individual_judges"] = {
            jid: r.winner for jid, r in jr.items()
        }
        return base

    def _weighted_vote(self, pid: str, jr: dict[str, PairwiseResult]) -> PairwiseResult:
        score_map: dict[str, float] = {"A": 0.0, "B": 0.0, "tie": 0.0}
        for jid, r in jr.items():
            w = self.primary_weight if jid == self.primary_judge_id else 1.0
            score_map[r.winner] += w

        winner = max(score_map, key=lambda k: score_map[k])
        top_two = sorted(score_map.values(), reverse=True)
        if top_two[0] - top_two[1] < 0.5:
            winner = "tie"

        base = list(jr.values())[0].model_copy()
        base.winner = winner
        base.judge_id = f"agg_weighted({len(jr)})"
        base.metadata["weighted_scores"] = score_map
        return base

    def _conservative(self, pid: str, jr: dict[str, PairwiseResult]) -> PairwiseResult:
        winners = set(r.winner for r in jr.values())

        if len(winners) == 1:
            winner = winners.pop()
        else:
            winner = "tie"

        base = list(jr.values())[0].model_copy()
        base.winner = winner
        base.judge_id = f"agg_conservative({len(jr)})"
        base.metadata["unanimous"] = len(winners) == 1
        return base


    def compute_agreement(
        self,
        results_by_judge: dict[str, list[PairwiseResult]],
    ) -> dict[str, Any]:
        """计算 Judge 间的一致性统计。"""
        grouped: dict[str, dict[str, str]] = {}
        for jid, results in results_by_judge.items():
            for r in results:
                grouped.setdefault(r.prompt_id, {})[jid] = r.winner

        total = len(grouped)
        unanimous = 0
        for pid, votes in grouped.items():
            if len(set(votes.values())) == 1:
                unanimous += 1

        judge_ids = list(results_by_judge.keys())
        pairwise_agree: dict[str, float] = {}
        for i in range(len(judge_ids)):
            for j in range(i + 1, len(judge_ids)):
                ji, jj = judge_ids[i], judge_ids[j]
                agree_cnt = 0
                pair_cnt = 0
                for pid in grouped:
                    if ji in grouped[pid] and jj in grouped[pid]:
                        pair_cnt += 1
                        if grouped[pid][ji] == grouped[pid][jj]:
                            agree_cnt += 1
                if pair_cnt:
                    pairwise_agree[f"{ji}_vs_{jj}"] = agree_cnt / pair_cnt

        return {
            "total_prompts": total,
            "unanimous_count": unanimous,
            "unanimous_rate": f"{unanimous / max(total, 1) * 100:.1f}%",
            "pairwise_agreement": pairwise_agree,
        }
