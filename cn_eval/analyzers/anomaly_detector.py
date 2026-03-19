"""
异常检测模块 — 自动标记异常样本。

异常类型：
  - length_outlier: 长度异常（超出 percentile 范围）
  - high_repetition: n-gram 重复率过高
  - judge_disagreement: 多 Judge 结论严重不一致
  - empty_response: 空回答
  - extreme_score: 极端分数
"""

from __future__ import annotations

import logging
from typing import Any

from cn_eval.data_loader.schema import (
    AnomalyFlag, ModelOutput, PairwiseResult, LongAnswerResult,
)
from cn_eval.utils.config import AnomalyConfig
from cn_eval.utils.text import count_chars, ngram_repetition_rate

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """异常样本检测器。"""

    def __init__(self, config: AnomalyConfig):
        self.config = config

    def detect_from_outputs(self, outputs: list[ModelOutput]) -> list[AnomalyFlag]:
        """从原始模型输出中检测异常。"""
        flags: list[AnomalyFlag] = []
        lengths = [count_chars(o.response) for o in outputs]

        if not lengths:
            return flags

        lo_pct, hi_pct = self.config.length_percentile
        sorted_lens = sorted(lengths)
        n = len(sorted_lens)
        lo_val = sorted_lens[int(n * lo_pct / 100)]
        hi_val = sorted_lens[int(min(n * hi_pct / 100, n - 1))]

        for output, length in zip(outputs, lengths):
            anomalies = []
            details: dict[str, Any] = {}

            if not output.response.strip():
                anomalies.append("empty_response")

            if length < lo_val or length > hi_val:
                anomalies.append("length_outlier")
                details["length"] = length
                details["expected_range"] = [lo_val, hi_val]

            rep_rate = ngram_repetition_rate(output.response, n=4, level="char")
            if rep_rate > self.config.ngram_repeat_threshold:
                anomalies.append("high_repetition")
                details["ngram_4_repeat_rate"] = round(rep_rate, 4)

            if anomalies:
                flags.append(AnomalyFlag(
                    prompt_id=output.prompt_id,
                    anomaly_types=anomalies,
                    details=details,
                ))

        logger.info(
            "[AnomalyDetector] 检出 %d / %d 条异常 (类型分布: %s)",
            len(flags), len(outputs), self._type_dist(flags),
        )
        return flags

    def detect_from_pairwise(
        self,
        results_by_judge: dict[str, list[PairwiseResult]],
    ) -> list[AnomalyFlag]:
        """从多 Judge 的 Pairwise 结果中检测 Judge 不一致。"""
        flags: list[AnomalyFlag] = []

        grouped: dict[str, dict[str, str]] = {}
        for jid, results in results_by_judge.items():
            for r in results:
                grouped.setdefault(r.prompt_id, {})[jid] = r.winner

        threshold = self.config.judge_disagreement_threshold
        for pid, votes in grouped.items():
            unique_winners = set(votes.values())
            if len(unique_winners) >= threshold:
                flags.append(AnomalyFlag(
                    prompt_id=pid,
                    anomaly_types=["judge_disagreement"],
                    details={
                        "votes": votes,
                        "unique_conclusions": len(unique_winners),
                    },
                ))

        if flags:
            logger.info("[AnomalyDetector] Judge 不一致: %d 条", len(flags))
        return flags

    def detect_from_long_answer(
        self,
        results: list[LongAnswerResult],
    ) -> list[AnomalyFlag]:
        """从长回答评测结果中检测异常。"""
        flags: list[AnomalyFlag] = []
        scores = [r.scores.mean() for r in results]

        if not scores:
            return flags

        mean_s = sum(scores) / len(scores)
        std_s = (sum((s - mean_s) ** 2 for s in scores) / len(scores)) ** 0.5
        threshold = self.config.common_anomaly_threshold

        for result, score in zip(results, scores):
            anomalies = []
            details: dict[str, Any] = {}

            if std_s > 0 and abs(score - mean_s) > threshold * std_s:
                anomalies.append("extreme_score")
                details["score"] = round(score, 3)
                details["z_score"] = round((score - mean_s) / std_s, 3)

            rep_stats = result.repetition_stats
            ngram_rates = rep_stats.get("ngram_rates", {})
            worst = max(ngram_rates.values()) if ngram_rates else 0
            if worst > self.config.ngram_repeat_threshold:
                anomalies.append("high_repetition")
                details["worst_ngram_rate"] = round(worst, 4)

            if result.style_stats.get("assistant_phrase_count", 0) > 3:
                anomalies.append("over_assistant_style")
                details["assistant_phrases"] = result.style_stats.get("assistant_phrases", [])

            if anomalies:
                flags.append(AnomalyFlag(
                    prompt_id=result.prompt_id,
                    anomaly_types=anomalies,
                    details=details,
                ))

        logger.info("[AnomalyDetector] 长回答异常: %d / %d 条", len(flags), len(results))
        return flags

    @staticmethod
    def _type_dist(flags: list[AnomalyFlag]) -> dict[str, int]:
        dist: dict[str, int] = {}
        for f in flags:
            for t in f.anomaly_types:
                dist[t] = dist.get(t, 0) + 1
        return dist
