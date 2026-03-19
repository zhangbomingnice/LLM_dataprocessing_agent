"""
基础统计 + 稳健统计模块。

覆盖：均值/中位数/分位数/标准差/trimmed mean/bootstrap CI/假设检验。
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

logger = logging.getLogger(__name__)


class StatsCalculator:
    """统计计算器。"""

    @staticmethod
    def basic(values: list[float]) -> dict[str, float]:
        """基础描述统计。"""
        if not values:
            return {}
        n = len(values)
        s = sorted(values)
        mean = sum(s) / n
        var = sum((x - mean) ** 2 for x in s) / n
        std = var ** 0.5
        return {
            "n": n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": s[0],
            "max": s[-1],
            "median": round(StatsCalculator._percentile(s, 0.5), 4),
            "p25": round(StatsCalculator._percentile(s, 0.25), 4),
            "p75": round(StatsCalculator._percentile(s, 0.75), 4),
            "p5": round(StatsCalculator._percentile(s, 0.05), 4),
            "p95": round(StatsCalculator._percentile(s, 0.95), 4),
        }

    @staticmethod
    def trimmed_mean(values: list[float], ratio: float = 0.1) -> float:
        """截尾均值 — 去掉两端 ratio 比例后取平均。"""
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)
        trim = int(n * ratio)
        if trim >= n // 2:
            trim = max(0, n // 2 - 1)
        trimmed = s[trim: n - trim] if trim > 0 else s
        return round(sum(trimmed) / len(trimmed), 4)

    @staticmethod
    def bootstrap_ci(
        values: list[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> dict[str, float]:
        """Bootstrap 置信区间。"""
        if not values or len(values) < 2:
            mean = values[0] if values else 0.0
            return {"mean": mean, "ci_lower": mean, "ci_upper": mean}

        rng = random.Random(seed)
        n = len(values)
        means = []
        for _ in range(n_bootstrap):
            sample = [rng.choice(values) for _ in range(n)]
            means.append(sum(sample) / n)

        means.sort()
        alpha = (1 - confidence) / 2
        lo_idx = int(n_bootstrap * alpha)
        hi_idx = int(n_bootstrap * (1 - alpha))

        return {
            "mean": round(sum(values) / n, 4),
            "ci_lower": round(means[lo_idx], 4),
            "ci_upper": round(means[min(hi_idx, len(means) - 1)], 4),
            "confidence": confidence,
            "n_bootstrap": n_bootstrap,
        }

    @staticmethod
    def wilcoxon_signed_rank(x: list[float], y: list[float]) -> dict[str, Any]:
        """
        Wilcoxon 符号秩检验 — 不依赖 scipy 的纯 Python 实现。
        
        检验配对样本的差异是否显著。
        """
        if len(x) != len(y):
            raise ValueError("两组数据长度不一致")

        diffs = [a - b for a, b in zip(x, y) if a != b]
        if not diffs:
            return {"statistic": 0, "p_value": 1.0, "significant": False, "n_pairs": len(x)}

        abs_diffs = [(abs(d), d) for d in diffs]
        abs_diffs.sort(key=lambda t: t[0])

        # 赋秩
        ranks = []
        i = 0
        while i < len(abs_diffs):
            j = i
            while j < len(abs_diffs) and abs_diffs[j][0] == abs_diffs[i][0]:
                j += 1
            avg_rank = (i + 1 + j) / 2
            for k in range(i, j):
                ranks.append((avg_rank, abs_diffs[k][1]))
            i = j

        w_plus = sum(r for r, d in ranks if d > 0)
        w_minus = sum(r for r, d in ranks if d < 0)
        w = min(w_plus, w_minus)
        n = len(diffs)

        # 正态近似 p 值
        mean_w = n * (n + 1) / 4
        std_w = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
        if std_w == 0:
            p_value = 1.0
        else:
            z = (w - mean_w) / std_w
            p_value = 2 * StatsCalculator._norm_cdf(-abs(z))

        return {
            "statistic": round(w, 4),
            "w_plus": round(w_plus, 4),
            "w_minus": round(w_minus, 4),
            "z_score": round((w - mean_w) / std_w if std_w else 0, 4),
            "p_value": round(p_value, 6),
            "significant_005": p_value < 0.05,
            "significant_001": p_value < 0.01,
            "n_pairs": n,
        }

    @staticmethod
    def sign_test(x: list[float], y: list[float]) -> dict[str, Any]:
        """符号检验。"""
        if len(x) != len(y):
            raise ValueError("两组数据长度不一致")

        pos = sum(1 for a, b in zip(x, y) if a > b)
        neg = sum(1 for a, b in zip(x, y) if a < b)
        n = pos + neg

        if n == 0:
            return {"statistic": 0, "p_value": 1.0, "significant": False}

        # 二项分布近似
        k = min(pos, neg)
        p_value = 0.0
        for i in range(k + 1):
            p_value += StatsCalculator._binom_pmf(n, i, 0.5)
        p_value *= 2
        p_value = min(p_value, 1.0)

        return {
            "positive": pos,
            "negative": neg,
            "ties": len(x) - n,
            "p_value": round(p_value, 6),
            "significant_005": p_value < 0.05,
        }

    @staticmethod
    def effect_size_cohens_d(x: list[float], y: list[float]) -> float:
        """Cohen's d 效应量。"""
        if not x or not y:
            return 0.0
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        var_x = sum((v - mean_x) ** 2 for v in x) / max(len(x) - 1, 1)
        var_y = sum((v - mean_y) ** 2 for v in y) / max(len(y) - 1, 1)
        pooled_std = ((var_x + var_y) / 2) ** 0.5
        if pooled_std == 0:
            return 0.0
        return round((mean_x - mean_y) / pooled_std, 4)

    # ── 辅助函数 ─────────────────────────────────────────

    @staticmethod
    def _percentile(sorted_vals: list[float], p: float) -> float:
        n = len(sorted_vals)
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    @staticmethod
    def _norm_cdf(z: float) -> float:
        """标准正态 CDF 近似 (Abramowitz & Stegun)。"""
        if z < -8:
            return 0.0
        if z > 8:
            return 1.0
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        sign = 1 if z >= 0 else -1
        z = abs(z) / (2 ** 0.5)
        t = 1.0 / (1.0 + p * z)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z)
        return 0.5 * (1.0 + sign * y)

    @staticmethod
    def _binom_pmf(n: int, k: int, p: float) -> float:
        """二项分布 PMF。"""
        if k > n or k < 0:
            return 0.0
        log_coeff = (
            StatsCalculator._log_factorial(n)
            - StatsCalculator._log_factorial(k)
            - StatsCalculator._log_factorial(n - k)
        )
        return math.exp(log_coeff + k * math.log(p) + (n - k) * math.log(1 - p))

    @staticmethod
    def _log_factorial(n: int) -> float:
        return sum(math.log(i) for i in range(1, n + 1)) if n > 0 else 0.0
