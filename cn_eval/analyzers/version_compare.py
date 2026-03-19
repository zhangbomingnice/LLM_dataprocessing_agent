"""
版本对比分析 — 多模型版本的全维度对比。
"""

from __future__ import annotations

import logging
from typing import Any

from cn_eval.data_loader.schema import PairwiseResult, LongAnswerResult, DimensionScores
from cn_eval.analyzers.basic_stats import StatsCalculator

logger = logging.getLogger(__name__)

DIMENSIONS = ["mode", "structure", "organization", "fluency", "non_repetition", "task_fit"]


class VersionComparer:
    """多版本对比分析器。"""

    def compare_pairwise(
        self,
        pairwise_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        从多组 Pairwise 结果中提取版本间对比。

        pairwise_results: {pair_key: {"results": [...], "summary": {...}}}
        """
        report: dict[str, Any] = {"pairs": {}}

        for pair_key, data in pairwise_results.items():
            summary = data.get("summary", {})
            results = data.get("results", [])

            pair_report = {
                "win_rate_a": summary.get("win_rate_a", 0),
                "win_rate_b": summary.get("win_rate_b", 0),
                "tie_rate": summary.get("tie_rate", 0),
                "total": summary.get("total", 0),
                "avg_scores_a": summary.get("avg_scores_a", {}),
                "avg_scores_b": summary.get("avg_scores_b", {}),
            }

            # 逐维度假设检验
            if results and isinstance(results[0], PairwiseResult):
                pair_report["hypothesis_tests"] = self._dimension_tests(results)
                pair_report["effect_sizes"] = self._effect_sizes(results)

            report["pairs"][pair_key] = pair_report

        return report

    def compare_long_answer(
        self,
        version_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        对比多个版本的长回答评测结果。

        version_results: {version: {"results": [...], "summary": {...}}}
        """
        report: dict[str, Any] = {"versions": {}, "dimension_comparison": {}}

        version_scores: dict[str, dict[str, list[float]]] = {}

        for version, data in version_results.items():
            results = data.get("results", [])
            summary = data.get("summary", {})
            report["versions"][version] = {
                "total": summary.get("total", 0),
                "mean_score": summary.get("mean_score", 0),
                "dimension_averages": summary.get("dimension_averages", {}),
                "quality_distribution": summary.get("quality_distribution", {}),
            }

            # 收集逐维度分数
            if results and isinstance(results[0], LongAnswerResult):
                for dim in DIMENSIONS:
                    version_scores.setdefault(dim, {})
                    version_scores[dim][version] = [
                        getattr(r.scores, dim) for r in results
                    ]

        # 跨版本维度对比
        versions = list(version_results.keys())
        if len(versions) == 2:
            v1, v2 = versions
            for dim in DIMENSIONS:
                scores_1 = version_scores.get(dim, {}).get(v1, [])
                scores_2 = version_scores.get(dim, {}).get(v2, [])
                if scores_1 and scores_2 and len(scores_1) == len(scores_2):
                    test = StatsCalculator.wilcoxon_signed_rank(scores_1, scores_2)
                    d = StatsCalculator.effect_size_cohens_d(scores_1, scores_2)
                    report["dimension_comparison"][dim] = {
                        f"{v1}_mean": round(sum(scores_1) / len(scores_1), 4),
                        f"{v2}_mean": round(sum(scores_2) / len(scores_2), 4),
                        "diff": round(sum(scores_1) / len(scores_1) - sum(scores_2) / len(scores_2), 4),
                        "cohens_d": d,
                        "wilcoxon_p": test["p_value"],
                        "significant": test["significant_005"],
                    }

        elif len(versions) > 2:
            for dim in DIMENSIONS:
                dim_report = {}
                for ver in versions:
                    scores = version_scores.get(dim, {}).get(ver, [])
                    if scores:
                        dim_report[ver] = StatsCalculator.basic(scores)
                report["dimension_comparison"][dim] = dim_report

        return report

    def summary_table(
        self,
        version_results: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """生成版本对比汇总表（适合输出为 CSV/表格）。"""
        rows = []
        for version, data in version_results.items():
            summary = data.get("summary", {})
            row = {
                "version": version,
                "total": summary.get("total", 0),
                "mean_score": summary.get("mean_score", 0),
            }
            dim_avgs = summary.get("dimension_averages", {})
            for dim in DIMENSIONS:
                row[f"avg_{dim}"] = dim_avgs.get(dim, 0)
            rows.append(row)
        return rows

    def _dimension_tests(self, results: list[PairwiseResult]) -> dict[str, Any]:
        """逐维度做配对假设检验。"""
        tests = {}
        for dim in DIMENSIONS:
            scores_a = [getattr(r.scores_a, dim) for r in results]
            scores_b = [getattr(r.scores_b, dim) for r in results]
            if any(a != b for a, b in zip(scores_a, scores_b)):
                tests[dim] = StatsCalculator.wilcoxon_signed_rank(scores_a, scores_b)
        return tests

    def _effect_sizes(self, results: list[PairwiseResult]) -> dict[str, float]:
        """逐维度计算效应量。"""
        effects = {}
        for dim in DIMENSIONS:
            scores_a = [getattr(r.scores_a, dim) for r in results]
            scores_b = [getattr(r.scores_b, dim) for r in results]
            effects[dim] = StatsCalculator.effect_size_cohens_d(scores_a, scores_b)
        return effects
