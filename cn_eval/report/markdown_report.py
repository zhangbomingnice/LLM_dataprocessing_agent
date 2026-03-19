"""
Markdown 报告生成器。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from cn_eval.data_loader.schema import DIMENSIONS, DIM_LABELS_ZH

logger = logging.getLogger(__name__)

_DIM_ZH = dict(zip(DIMENSIONS, DIM_LABELS_ZH))


class MarkdownReporter:
    """生成 Markdown 格式的评测报告。"""

    def generate(
        self,
        results: dict[str, Any],
        config_summary: dict[str, Any] | None = None,
        output_path: str | Path = "outputs/report.md",
    ) -> str:
        lines: list[str] = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines.append("# 中文长回答 SFT 评测报告")
        lines.append("")
        lines.append(f"> 生成时间: {now}")
        lines.append("")

        if config_summary:
            lines.append("## 评测配置")
            lines.append("")
            lines.append("| 参数 | 值 |")
            lines.append("|------|-----|")
            for k, v in config_summary.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        for mode, data in results.items():
            lines.append(f"## {self._mode_title(mode)}")
            lines.append("")

            if mode == "pairwise":
                lines.extend(self._render_pairwise(data))
            elif mode == "single":
                lines.extend(self._render_single(data))
            else:
                lines.extend(self._render_generic(data))

            lines.append("")

        lines.append("---")
        lines.append("*由 cn_eval 统一 LLM Judge 自动生成*")

        content = "\n".join(lines)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        logger.info("[Report] Markdown: %s", output_path)

        return content

    def _mode_title(self, mode: str) -> str:
        titles = {
            "pairwise": "配对对比评测 (Pairwise)",
            "single": "单模型评测 (Single)",
        }
        return titles.get(mode, mode)

    def _render_pairwise(self, data: dict) -> list[str]:
        lines: list[str] = []
        summary = data.get("summary", {})

        if not summary:
            for pair_key, pair_data in data.items():
                if isinstance(pair_data, dict) and "summary" in pair_data:
                    lines.append(f"### {pair_key}")
                    lines.append("")
                    lines.extend(self._render_pairwise_single(pair_data))
            return lines

        lines.extend(self._render_pairwise_single(data))
        return lines

    def _render_pairwise_single(self, data: dict) -> list[str]:
        lines: list[str] = []
        summary = data.get("summary", {})

        lines.append("### 胜率统计")
        lines.append("")
        lines.append("| 结果 | 比例 |")
        lines.append("|------|------|")
        lines.append(f"| Baseline 胜 | {summary.get('win_rate_a', 0):.1%} |")
        lines.append(f"| Candidate 胜 | {summary.get('win_rate_b', 0):.1%} |")
        lines.append(f"| 平局 | {summary.get('tie_rate', 0):.1%} |")
        lines.append(f"| 总计 | {summary.get('total', 0)} 条 |")
        lines.append("")

        consistency = summary.get("consistency", {})
        if consistency:
            lines.append(f"**一致性**: 平均胜者一致率 {consistency.get('avg_winner_agreement', 0):.1%}")
            lines.append("")

        scores_a = summary.get("avg_scores_a", {})
        scores_b = summary.get("avg_scores_b", {})
        if scores_a and scores_b:
            lines.append("### 维度平均分对比")
            lines.append("")
            lines.append("| 维度 | Baseline | Candidate | 差值 |")
            lines.append("|------|----------|-----------|------|")
            for dim in DIMENSIONS:
                a = scores_a.get(dim, 0)
                b = scores_b.get(dim, 0)
                diff = b - a
                arrow = "↑" if diff > 0.1 else ("↓" if diff < -0.1 else "≈")
                zh = _DIM_ZH.get(dim, dim)
                lines.append(f"| {zh} | {a:.3f} | {b:.3f} | {diff:+.3f} {arrow} |")
            lines.append("")

        cat_stats = summary.get("by_category", {})
        if cat_stats:
            lines.append("### 分类统计")
            lines.append("")
            lines.append("| 分类 | 总数 | A 胜 | B 胜 | 平局 |")
            lines.append("|------|------|------|------|------|")
            for cat, stats in cat_stats.items():
                lines.append(
                    f"| {cat} | {stats['total']} | {stats.get('A', 0)} | "
                    f"{stats.get('B', 0)} | {stats.get('tie', 0)} |"
                )
            lines.append("")

        return lines

    def _render_single(self, data: dict) -> list[str]:
        lines: list[str] = []

        for version, vdata in data.items():
            if not isinstance(vdata, dict):
                continue
            summary = vdata.get("summary", {})
            lines.append(f"### 模型: {version}")
            lines.append("")
            lines.append(f"- 总数: {summary.get('total', 0)}")
            lines.append(f"- 平均分: **{summary.get('mean_score', 0):.3f}**")
            lines.append("")

            dim_avgs = summary.get("dimension_averages", {})
            if dim_avgs:
                lines.append("| 维度 | 平均分 |")
                lines.append("|------|--------|")
                for dim in DIMENSIONS:
                    zh = _DIM_ZH.get(dim, dim)
                    lines.append(f"| {zh} | {dim_avgs.get(dim, 0):.3f} |")
                lines.append("")

            consistency = summary.get("consistency", {})
            if consistency:
                lines.append(
                    f"**一致性**: 不确定率 {consistency.get('uncertain_rate', 0):.1%}, "
                    f"平均最大标准差 {consistency.get('avg_max_std', 0):.3f}"
                )
                lines.append("")

            qual_dist = summary.get("quality_distribution", {})
            if qual_dist:
                lines.append("**质量分布:**")
                for bucket, count in qual_dist.items():
                    lines.append(f"- {bucket}: {count}")
                lines.append("")

            rule_flags = summary.get("rule_flags", {})
            if rule_flags:
                lines.append("**规则检出:**")
                for flag, count in rule_flags.items():
                    lines.append(f"- {flag}: {count}")
                lines.append("")

        return lines

    def _render_generic(self, data: dict) -> list[str]:
        lines: list[str] = []
        if isinstance(data, dict):
            for k, v in data.items():
                lines.append(f"- **{k}**: {v}")
        lines.append("")
        return lines
