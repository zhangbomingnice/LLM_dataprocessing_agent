"""
Markdown 报告生成器。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
            elif mode == "long_answer":
                lines.extend(self._render_long_answer(data))
            elif mode == "if_eval":
                lines.extend(self._render_if_eval(data))
            elif mode == "benchmark":
                lines.extend(self._render_benchmark(data))
            else:
                lines.extend(self._render_generic(data))

            lines.append("")

        lines.append("---")
        lines.append("*由 cn_eval 自动生成*")

        content = "\n".join(lines)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        logger.info("[Report] Markdown 报告已保存: %s", output_path)

        return content

    def _mode_title(self, mode: str) -> str:
        titles = {
            "pairwise": "配对对比评测 (Pairwise)",
            "long_answer": "长回答专项评测 (Long-Answer)",
            "if_eval": "指令遵循评测 (IF-Eval)",
            "benchmark": "基准评测 (Benchmark)",
        }
        return titles.get(mode, mode)

    def _render_pairwise(self, data: dict) -> list[str]:
        lines = []
        summary = data.get("summary", {})

        if not summary:
            # 可能是多对比较，嵌套结构
            for pair_key, pair_data in data.items():
                if isinstance(pair_data, dict) and "summary" in pair_data:
                    lines.append(f"### {pair_key}")
                    lines.append("")
                    lines.extend(self._render_pairwise_single(pair_data))
            return lines

        lines.extend(self._render_pairwise_single(data))
        return lines

    def _render_pairwise_single(self, data: dict) -> list[str]:
        lines = []
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

        # 维度对比
        scores_a = summary.get("avg_scores_a", {})
        scores_b = summary.get("avg_scores_b", {})
        if scores_a and scores_b:
            lines.append("### 维度平均分对比")
            lines.append("")
            lines.append("| 维度 | Baseline | Candidate | 差值 |")
            lines.append("|------|----------|-----------|------|")
            for dim in scores_a:
                a = scores_a.get(dim, 0)
                b = scores_b.get(dim, 0)
                diff = b - a
                arrow = "↑" if diff > 0.1 else ("↓" if diff < -0.1 else "≈")
                lines.append(f"| {dim} | {a:.3f} | {b:.3f} | {diff:+.3f} {arrow} |")
            lines.append("")

        # 分类统计
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

    def _render_long_answer(self, data: dict) -> list[str]:
        lines = []

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
                for dim, val in dim_avgs.items():
                    lines.append(f"| {dim} | {val:.3f} |")
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

    def _render_if_eval(self, data: dict) -> list[str]:
        lines = []

        for version, vdata in data.items():
            if not isinstance(vdata, dict):
                continue
            summary = vdata.get("summary", {})
            lines.append(f"### 模型: {version}")
            lines.append("")
            lines.append(f"- 总数: {summary.get('total', 0)}")
            lines.append(f"- 通过: {summary.get('passed', 0)}")
            lines.append(f"- 失败: {summary.get('failed', 0)}")
            lines.append(f"- 通过率: **{summary.get('pass_rate', 0):.1%}**")
            lines.append("")

            check_stats = summary.get("check_stats", {})
            if check_stats:
                lines.append("| 检查项 | 总数 | 通过 | 通过率 |")
                lines.append("|--------|------|------|--------|")
                for check, stats in check_stats.items():
                    t = stats["total"]
                    p = stats["passed"]
                    rate = p / t if t else 0
                    lines.append(f"| {check} | {t} | {p} | {rate:.1%} |")
                lines.append("")

        return lines

    def _render_benchmark(self, data: dict) -> list[str]:
        lines = []

        for version, vdata in data.items():
            if not isinstance(vdata, dict):
                continue
            summary = vdata.get("summary", {})
            lines.append(f"### 模型: {version}")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for k, v in summary.items():
                if k == "total":
                    lines.append(f"| 总数 | {v} |")
                elif isinstance(v, float):
                    lines.append(f"| {k} | {v:.4f} |")
                else:
                    lines.append(f"| {k} | {v} |")
            lines.append("")

        return lines

    def _render_generic(self, data: dict) -> list[str]:
        lines = []
        if isinstance(data, dict):
            for k, v in data.items():
                lines.append(f"- **{k}**: {v}")
        lines.append("")
        return lines
