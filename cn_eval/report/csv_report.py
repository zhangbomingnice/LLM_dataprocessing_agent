"""
CSV 报告生成器 — 输出结构化表格数据。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from cn_eval.data_loader.schema import PairwiseResult, LongAnswerResult, IFResult

logger = logging.getLogger(__name__)


class CSVReporter:
    """生成 CSV 格式的评测结果。"""

    def export_pairwise(
        self,
        results: list[PairwiseResult],
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "prompt_id", "model_a", "model_b", "winner",
            "a_mode", "a_structure", "a_organization", "a_fluency", "a_non_repetition", "a_task_fit", "a_mean",
            "b_mode", "b_structure", "b_organization", "b_fluency", "b_non_repetition", "b_task_fit", "b_mean",
            "judge_id", "reasoning",
        ]

        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "prompt_id": r.prompt_id,
                    "model_a": r.model_a,
                    "model_b": r.model_b,
                    "winner": r.winner,
                    "a_mode": r.scores_a.mode,
                    "a_structure": r.scores_a.structure,
                    "a_organization": r.scores_a.organization,
                    "a_fluency": r.scores_a.fluency,
                    "a_non_repetition": r.scores_a.non_repetition,
                    "a_task_fit": r.scores_a.task_fit,
                    "a_mean": round(r.scores_a.mean(), 3),
                    "b_mode": r.scores_b.mode,
                    "b_structure": r.scores_b.structure,
                    "b_organization": r.scores_b.organization,
                    "b_fluency": r.scores_b.fluency,
                    "b_non_repetition": r.scores_b.non_repetition,
                    "b_task_fit": r.scores_b.task_fit,
                    "b_mean": round(r.scores_b.mean(), 3),
                    "judge_id": r.judge_id,
                    "reasoning": r.reasoning[:200],
                })

        logger.info("[CSVReport] Pairwise CSV: %s (%d 条)", path, len(results))

    def export_long_answer(
        self,
        results: list[LongAnswerResult],
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "prompt_id", "model_version",
            "mode", "structure", "organization", "fluency", "non_repetition", "task_fit", "mean",
            "repetition_worst", "template_count", "assistant_count",
            "paragraph_count", "sentence_count",
        ]

        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                ngram = r.repetition_stats.get("ngram_rates", {})
                worst_rep = max(ngram.values()) if ngram else 0
                writer.writerow({
                    "prompt_id": r.prompt_id,
                    "model_version": r.model_version,
                    "mode": r.scores.mode,
                    "structure": r.scores.structure,
                    "organization": r.scores.organization,
                    "fluency": r.scores.fluency,
                    "non_repetition": r.scores.non_repetition,
                    "task_fit": r.scores.task_fit,
                    "mean": round(r.scores.mean(), 3),
                    "repetition_worst": round(worst_rep, 4),
                    "template_count": r.style_stats.get("template_ending_count", 0),
                    "assistant_count": r.style_stats.get("assistant_phrase_count", 0),
                    "paragraph_count": r.structure_stats.get("paragraph_count", 0),
                    "sentence_count": r.structure_stats.get("sentence_count", 0),
                })

        logger.info("[CSVReport] Long-Answer CSV: %s (%d 条)", path, len(results))

    def export_if_eval(
        self,
        results: list[IFResult],
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        all_checks = set()
        for r in results:
            all_checks.update(r.checks.keys())
        sorted_checks = sorted(all_checks)

        fieldnames = ["prompt_id", "model_version", "passed"] + sorted_checks + ["details"]

        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {
                    "prompt_id": r.prompt_id,
                    "model_version": r.model_version,
                    "passed": r.passed,
                    "details": r.details[:200],
                }
                for check in sorted_checks:
                    row[check] = r.checks.get(check, "")
                writer.writerow(row)

        logger.info("[CSVReport] IF-Eval CSV: %s (%d 条)", path, len(results))

    def export_anomalies(
        self,
        anomalies: list[dict],
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["prompt_id", "anomaly_types", "details"]

        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for a in anomalies:
                types = a.get("anomaly_types", [])
                if hasattr(a, "anomaly_types"):
                    types = a.anomaly_types
                writer.writerow({
                    "prompt_id": getattr(a, "prompt_id", a.get("prompt_id", "")),
                    "anomaly_types": "; ".join(types if isinstance(types, list) else [str(types)]),
                    "details": str(getattr(a, "details", a.get("details", "")))[:300],
                })

        logger.info("[CSVReport] Anomalies CSV: %s (%d 条)", path, len(anomalies))

    def export_version_table(
        self,
        rows: list[dict[str, Any]],
        output_path: str | Path,
    ) -> None:
        if not rows:
            return
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(rows[0].keys())
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("[CSVReport] Version table CSV: %s (%d 条)", path, len(rows))
