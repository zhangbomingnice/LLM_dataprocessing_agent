"""
Pairwise Evaluator — 配对对比评测。

流程：Aligner 对齐 → BlindAB 评审 → 多 Judge 聚合 → 多轮聚合 → 统计
"""

from __future__ import annotations

import logging
from typing import Any

from cn_eval.data_loader.schema import (
    Prompt, ModelOutput, AlignedPair, PairwiseResult, DimensionScores,
)
from cn_eval.aligner.prompt_aligner import PromptAligner
from cn_eval.aligner.multi_round import MultiRoundAggregator
from cn_eval.judges.llm_judge import LLMJudge
from cn_eval.judges.rule_judge import RuleJudge
from cn_eval.judges.blind_ab import BlindABJudge
from cn_eval.judges.aggregator import JudgeAggregator
from cn_eval.utils.llm_client import LLMClient
from cn_eval.utils.config import EvalConfig
from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class PairwiseEvaluator(BaseEvaluator):
    """配对对比评测器。"""

    name = "pairwise"

    def __init__(self, config: EvalConfig, client: LLMClient):
        super().__init__(config)
        self.client = client

    async def run(
        self,
        prompts: list[Prompt],
        baseline_outputs: list[ModelOutput],
        candidate_outputs: list[ModelOutput],
        **kwargs: Any,
    ) -> dict[str, Any]:
        # 1. 对齐
        aligner = PromptAligner(prompts)
        pairs = aligner.align_pair(
            baseline_outputs, candidate_outputs,
            baseline_version=self.config.baseline,
            candidate_version=kwargs.get("candidate_version", "candidate"),
        )

        if not pairs:
            logger.warning("[PairwiseEval] 无可对齐数据")
            return {"results": [], "summary": {}}

        # 2. 构建 Judge
        llm_judge = LLMJudge(
            client=self.client,
            prompt_template_path=self.config.judge.prompt_template,
            judge_id="llm_primary",
            temperature=self.config.judge.temperature,
        )

        results_by_judge: dict[str, list[PairwiseResult]] = {}

        # 2a. LLM Judge + BlindAB
        if self.config.judge.blind_ab:
            blind = BlindABJudge(
                judge=llm_judge,
                swap_ratio=self.config.judge.swap_ratio,
            )
            llm_results = await blind.evaluate_batch(
                pairs, concurrency=self.config.judge.concurrency,
            )
        else:
            llm_results = await self._direct_judge(llm_judge, pairs)

        results_by_judge["llm_primary"] = llm_results

        # 2b. Rule Judge
        rule_judge = RuleJudge(
            ngram_repeat_threshold=self.config.anomaly.ngram_repeat_threshold,
        )
        rule_items = [
            {
                "prompt_id": p.prompt_id,
                "prompt_text": p.prompt_text,
                "response_a": p.baseline_response,
                "response_b": p.candidate_response,
            }
            for p in pairs
        ]
        rule_raw = await rule_judge.judge_batch(rule_items)
        rule_results = []
        for raw, pair in zip(rule_raw, pairs):
            rule_results.append(PairwiseResult(
                prompt_id=pair.prompt_id,
                model_a=pair.baseline_version,
                model_b=pair.candidate_version,
                winner=raw.get("winner", "tie"),
                judge_id="rule",
                metadata=raw,
            ))
        results_by_judge["rule"] = rule_results

        # 3. 聚合
        aggregator = JudgeAggregator(
            strategy=self.config.judge.aggregation,
            primary_judge_id="llm_primary",
        )
        aggregated = aggregator.aggregate(results_by_judge)

        # 4. 统计
        summary = self._compute_summary(aggregated, pairs)
        summary["judge_agreement"] = aggregator.compute_agreement(results_by_judge)

        logger.info(
            "[PairwiseEval] 完成: %d 对, 胜率 A=%.1f%% B=%.1f%% tie=%.1f%%",
            len(aggregated),
            summary.get("win_rate_a", 0) * 100,
            summary.get("win_rate_b", 0) * 100,
            summary.get("tie_rate", 0) * 100,
        )

        return {"results": aggregated, "summary": summary, "pairs": pairs}

    async def _direct_judge(
        self, judge: LLMJudge, pairs: list[AlignedPair],
    ) -> list[PairwiseResult]:
        """不走 BlindAB 的直接评审。"""
        items = [
            {
                "prompt_id": p.prompt_id,
                "prompt_text": p.prompt_text,
                "response_a": p.baseline_response,
                "response_b": p.candidate_response,
            }
            for p in pairs
        ]
        raw_results = await judge.judge_batch(items, concurrency=self.config.judge.concurrency)

        results = []
        for raw, pair in zip(raw_results, pairs):
            results.append(PairwiseResult(
                prompt_id=pair.prompt_id,
                model_a=pair.baseline_version,
                model_b=pair.candidate_version,
                winner=raw.get("winner", "tie"),
                judge_id=raw.get("judge_id", "llm_primary"),
                scores_a=LLMJudge.parse_scores({"scores": raw.get("scores_a", {})}),
                scores_b=LLMJudge.parse_scores({"scores": raw.get("scores_b", {})}),
                reasoning=raw.get("reasoning", ""),
            ))
        return results

    def _compute_summary(
        self, results: list[PairwiseResult], pairs: list[AlignedPair],
    ) -> dict[str, Any]:
        total = len(results) or 1
        wins_a = sum(1 for r in results if r.winner == "A")
        wins_b = sum(1 for r in results if r.winner == "B")
        ties = sum(1 for r in results if r.winner == "tie")

        # 按 category 分组统计
        cat_stats: dict[str, dict] = {}
        pair_map = {p.prompt_id: p.category for p in pairs}
        for r in results:
            cat = pair_map.get(r.prompt_id, "unknown")
            if cat not in cat_stats:
                cat_stats[cat] = {"total": 0, "A": 0, "B": 0, "tie": 0}
            cat_stats[cat]["total"] += 1
            cat_stats[cat][r.winner if r.winner in ("A", "B") else "tie"] += 1

        # 维度平均分
        dims_a = self._avg_dimension_scores([r.scores_a for r in results])
        dims_b = self._avg_dimension_scores([r.scores_b for r in results])

        return {
            "total": total,
            "win_rate_a": wins_a / total,
            "win_rate_b": wins_b / total,
            "tie_rate": ties / total,
            "by_category": cat_stats,
            "avg_scores_a": dims_a,
            "avg_scores_b": dims_b,
        }

    @staticmethod
    def _avg_dimension_scores(scores: list[DimensionScores]) -> dict[str, float]:
        if not scores:
            return {}
        n = len(scores)
        return {
            "mode": sum(s.mode for s in scores) / n,
            "structure": sum(s.structure for s in scores) / n,
            "organization": sum(s.organization for s in scores) / n,
            "fluency": sum(s.fluency for s in scores) / n,
            "non_repetition": sum(s.non_repetition for s in scores) / n,
            "task_fit": sum(s.task_fit for s in scores) / n,
        }
