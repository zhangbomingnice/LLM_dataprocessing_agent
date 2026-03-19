"""
图表生成器 — 使用 matplotlib 生成评测可视化图表。

如果 matplotlib 不可用则优雅降级。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cn_eval.data_loader.schema import DIMENSIONS, DIM_LABELS_ZH

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.info("[ChartGen] matplotlib 不可用，图表功能已禁用")


def _setup_chinese_font():
    if not HAS_MPL:
        return
    chinese_fonts = ["SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC"]
    for font in chinese_fonts:
        try:
            fm.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


class ChartGenerator:
    """评测图表生成器。"""

    def __init__(self, output_dir: str | Path = "outputs/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if HAS_MPL:
            _setup_chinese_font()

    def radar_chart(
        self,
        version_scores: dict[str, dict[str, float]],
        title: str = "维度对比雷达图",
        filename: str = "radar.png",
    ) -> str | None:
        if not HAS_MPL or not version_scores:
            return None

        import numpy as np

        labels = DIM_LABELS_ZH
        n = len(labels)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = plt.cm.Set2.colors

        for idx, (version, scores) in enumerate(version_scores.items()):
            values = [scores.get(dim, 0) for dim in DIMENSIONS]
            values += values[:1]
            color = colors[idx % len(colors)]
            ax.plot(angles, values, "o-", linewidth=2, label=version, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        path = self.output_dir / filename
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[ChartGen] 雷达图: %s", path)
        return str(path)

    def bar_chart(
        self,
        version_scores: dict[str, dict[str, float]],
        title: str = "维度分数对比",
        filename: str = "bar_compare.png",
    ) -> str | None:
        if not HAS_MPL or not version_scores:
            return None

        import numpy as np

        versions = list(version_scores.keys())
        n_dims = len(DIMENSIONS)
        x = np.arange(n_dims)
        width = 0.8 / max(len(versions), 1)
        colors = plt.cm.Set2.colors

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, version in enumerate(versions):
            scores = version_scores[version]
            values = [scores.get(dim, 0) for dim in DIMENSIONS]
            offset = (idx - len(versions) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=version, color=colors[idx % len(colors)])
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("维度")
        ax.set_ylabel("分数")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(DIM_LABELS_ZH)
        ax.set_ylim(0, 5.5)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        path = self.output_dir / filename
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[ChartGen] 柱状图: %s", path)
        return str(path)

    def win_rate_pie(
        self,
        win_rate_a: float,
        win_rate_b: float,
        tie_rate: float,
        label_a: str = "Baseline",
        label_b: str = "Candidate",
        title: str = "胜率分布",
        filename: str = "winrate_pie.png",
    ) -> str | None:
        if not HAS_MPL:
            return None

        fig, ax = plt.subplots(figsize=(8, 8))
        sizes = [win_rate_a, win_rate_b, tie_rate]
        labels = [f"{label_a}\n{win_rate_a:.1%}", f"{label_b}\n{win_rate_b:.1%}", f"平局\n{tie_rate:.1%}"]
        colors = ["#ff9999", "#66b3ff", "#99ff99"]
        explode = (0.05, 0.05, 0)

        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct="", startangle=90, textprops={"fontsize": 13})
        ax.set_title(title, fontsize=14, fontweight="bold")

        path = self.output_dir / filename
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[ChartGen] 饼图: %s", path)
        return str(path)

    def score_distribution(
        self,
        scores: list[float],
        title: str = "分数分布",
        filename: str = "score_dist.png",
    ) -> str | None:
        if not HAS_MPL or not scores:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(scores, bins=20, edgecolor="black", alpha=0.7, color="#4CAF50")
        mean = sum(scores) / len(scores)
        ax.axvline(mean, color="red", linestyle="--", label=f"均值 = {mean:.2f}")
        ax.set_xlabel("分数")
        ax.set_ylabel("频次")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        path = self.output_dir / filename
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("[ChartGen] 分布图: %s", path)
        return str(path)
