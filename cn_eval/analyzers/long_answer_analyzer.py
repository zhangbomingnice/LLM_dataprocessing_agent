"""
长回答深度分析模块。

分析维度：
  - 重复模式（逐句/逐段/前后半段）
  - 结构质量（段落分布/标题层次/列表密度）
  - 风格诊断（模板化/助手化/语气词密度）
  - 信息密度（去重后有效字符占比）
"""

from __future__ import annotations

import logging
import re
from typing import Any

from cn_eval.data_loader.schema import LongAnswerResult
from cn_eval.utils.text import (
    split_sentences, split_paragraphs, count_chars,
    ngram_repetition_rate, char_ngrams,
    detect_template_endings, detect_assistant_phrases,
)

logger = logging.getLogger(__name__)


class LongAnswerAnalyzer:
    """长回答深度分析器。"""

    def analyze_batch(self, results: list[LongAnswerResult]) -> dict[str, Any]:
        """批量分析，输出全局统计和逐条详情。"""
        if not results:
            return {}

        item_reports = [self.analyze_single(r) for r in results]

        # 全局聚合
        global_report = self._aggregate(item_reports)
        global_report["items"] = item_reports
        global_report["total"] = len(results)

        logger.info("[LongAnswerAnalyzer] 分析完成: %d 条", len(results))
        return global_report

    def analyze_single(self, result: LongAnswerResult) -> dict[str, Any]:
        """单条深度分析。"""
        # 从已有的评测结果中取 response（通过 metadata 或重建）
        rep = result.repetition_stats
        struct = result.structure_stats
        style = result.style_stats

        report: dict[str, Any] = {
            "prompt_id": result.prompt_id,
            "model_version": result.model_version,
            "overall_score": round(result.scores.mean(), 3),
            "dimension_scores": result.scores.to_dict(),
        }

        # 重复分析摘要
        ngram_rates = rep.get("ngram_rates", {})
        worst_ngram_key = max(ngram_rates, key=ngram_rates.get) if ngram_rates else ""
        report["repetition"] = {
            "worst_ngram": worst_ngram_key,
            "worst_rate": ngram_rates.get(worst_ngram_key, 0),
            "half_split_repeat": rep.get("half_split_repeat", 0),
            "severity": self._repetition_severity(ngram_rates),
        }

        # 结构分析摘要
        report["structure"] = {
            "paragraph_count": struct.get("paragraph_count", 0),
            "avg_para_length": struct.get("avg_paragraph_length", 0),
            "length_balance": self._length_balance(struct),
            "has_headings": struct.get("heading_count", 0) > 0,
            "list_density": struct.get("list_item_count", 0),
            "quality": self._structure_quality(struct),
        }

        # 风格诊断
        report["style"] = {
            "template_endings": style.get("template_endings", []),
            "assistant_phrases": style.get("assistant_phrases", []),
            "is_template_heavy": style.get("template_ending_count", 0) > 1,
            "is_assistant_heavy": style.get("assistant_phrase_count", 0) > 2,
            "diagnosis": self._style_diagnosis(style),
        }

        # 综合诊断
        issues = []
        if report["repetition"]["severity"] in ("high", "critical"):
            issues.append("重复内容过多")
        if report["style"]["is_template_heavy"]:
            issues.append("模板化结尾严重")
        if report["style"]["is_assistant_heavy"]:
            issues.append("助手化用语过多")
        if report["structure"]["quality"] == "poor":
            issues.append("结构质量差")

        report["issues"] = issues
        report["issue_count"] = len(issues)

        return report

    def _repetition_severity(self, ngram_rates: dict) -> str:
        if not ngram_rates:
            return "normal"
        worst = max(ngram_rates.values())
        if worst > 0.5:
            return "critical"
        if worst > 0.3:
            return "high"
        if worst > 0.15:
            return "moderate"
        return "normal"

    def _length_balance(self, struct: dict) -> str:
        std = struct.get("paragraph_length_std", 0)
        avg = struct.get("avg_paragraph_length", 1)
        if avg == 0:
            return "unknown"
        cv = std / avg
        if cv > 1.5:
            return "unbalanced"
        if cv > 0.8:
            return "moderate"
        return "balanced"

    def _structure_quality(self, struct: dict) -> str:
        para_count = struct.get("paragraph_count", 0)
        sent_count = struct.get("sentence_count", 0)

        if para_count <= 1 and sent_count > 5:
            return "poor"
        if para_count >= 3 and sent_count >= 5:
            return "good"
        return "acceptable"

    def _style_diagnosis(self, style: dict) -> str:
        template_n = style.get("template_ending_count", 0)
        assistant_n = style.get("assistant_phrase_count", 0)

        if template_n > 2 and assistant_n > 3:
            return "严重模板化+助手化"
        if template_n > 1:
            return "模板化倾向"
        if assistant_n > 2:
            return "助手化倾向"
        return "风格正常"

    def _aggregate(self, items: list[dict]) -> dict[str, Any]:
        """全局聚合分析。"""
        if not items:
            return {}

        scores = [it["overall_score"] for it in items]
        n = len(scores)

        # 问题分布
        issue_dist: dict[str, int] = {}
        for it in items:
            for issue in it.get("issues", []):
                issue_dist[issue] = issue_dist.get(issue, 0) + 1

        # 严重程度分布
        rep_severity: dict[str, int] = {}
        for it in items:
            sev = it.get("repetition", {}).get("severity", "unknown")
            rep_severity[sev] = rep_severity.get(sev, 0) + 1

        style_diag: dict[str, int] = {}
        for it in items:
            diag = it.get("style", {}).get("diagnosis", "unknown")
            style_diag[diag] = style_diag.get(diag, 0) + 1

        struct_q: dict[str, int] = {}
        for it in items:
            q = it.get("structure", {}).get("quality", "unknown")
            struct_q[q] = struct_q.get(q, 0) + 1

        return {
            "score_mean": round(sum(scores) / n, 3),
            "score_std": round((sum((s - sum(scores) / n) ** 2 for s in scores) / n) ** 0.5, 3),
            "issue_distribution": issue_dist,
            "issue_rate": round(sum(1 for it in items if it["issue_count"] > 0) / n, 3),
            "repetition_severity_dist": rep_severity,
            "style_diagnosis_dist": style_diag,
            "structure_quality_dist": struct_q,
        }
