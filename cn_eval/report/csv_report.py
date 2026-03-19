"""
CSV 报告生成器 — 输出结构化表格数据。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from cn_eval.data_loader.schema import PairwiseResult, EvalResult, DIMENSIONS

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

        a_dims = [f"a_{d}" for d in DIMENSIONS]
        b_dims = [f"b_{d}" for d in DIMENSIONS]
        fieldnames = [
            "prompt_id", "model_a", "model_b", "winner",
            *a_dims, "a_mean",
            *b_dims, "b_mean",
            "winner_agreement", "judge_id", "reasoning",
        ]

        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row: dict[str, Any] = {
                    "prompt_id": r.prompt_id,
                    "model_a": r.model_a,
                    "model_b": r.model_b,
                    "winner": r.winner,
                    "a_mean": round(r.scores_a.mean(), 3),
                    "b_mean": round(r.scores_b.mean(), 3),
                    "winner_agreement": r.consistency.get("winner_agreement", ""),
                    "judge_id": r.judge_id,
                    "reasoning": r.reasoning[:200],
                }
                for d in DIMENSIONS:
                    row[f"a_{d}"] = getattr(r.scores_a, d)
                    row[f"b_{d}"] = getattr(r.scores_b, d)
                writer.writerow(row)

        logger.info("[CSVReport] Pairwise: %s (%d 条)", path, len(results))

    def export_single(
        self,
        results: list[EvalResult],
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "prompt_id", "model_version",
            *DIMENSIONS, "mean",
            "uncertain", "max_std",
            "repetition_worst", "template_count", "assistant_count",
            "paragraph_count", "sentence_count",
        ]

        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                pre = r.pre_analysis
                ngram = pre.get("repetition", {}).get("ngram_rates", {})
                worst_rep = max(ngram.values()) if ngram else 0
                row: dict[str, Any] = {
                    "prompt_id": r.prompt_id,
                    "model_version": r.model_version,
                    "mean": round(r.scores.mean(), 3),
                    "uncertain": r.consistency.get("uncertain", False),
                    "max_std": r.consistency.get("max_std", 0),
                    "repetition_worst": round(worst_rep, 4),
                    "template_count": pre.get("style", {}).get("template_ending_count", 0),
                    "assistant_count": pre.get("style", {}).get("assistant_phrase_count", 0),
                    "paragraph_count": pre.get("structure", {}).get("paragraph_count", 0),
                    "sentence_count": pre.get("structure", {}).get("sentence_count", 0),
                }
                for d in DIMENSIONS:
                    row[d] = getattr(r.scores, d)
                writer.writerow(row)

        logger.info("[CSVReport] Single: %s (%d 条)", path, len(results))

    def export_anomalies(
        self,
        anomalies: list,
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["prompt_id", "anomaly_types", "details"]

        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for a in anomalies:
                types = getattr(a, "anomaly_types", a.get("anomaly_types", []))
                writer.writerow({
                    "prompt_id": getattr(a, "prompt_id", a.get("prompt_id", "")),
                    "anomaly_types": "; ".join(types if isinstance(types, list) else [str(types)]),
                    "details": str(getattr(a, "details", a.get("details", "")))[:300],
                })

        logger.info("[CSVReport] Anomalies: %s (%d 条)", path, len(anomalies))

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

        logger.info("[CSVReport] Version table: %s (%d 条)", path, len(rows))
