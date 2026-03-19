"""
统一 LLM Judge 评测器。

核心流程：
  1. 规则预分析 → 生成客观数据报告（重复率、结构统计、AI 痕迹）
  2. 预分析报告注入 LLM prompt → LLM 评审时参考客观数据
  3. 多轮评审（K=3）→ 每轮微调温度 + 奇数轮交换 AB 位置
  4. 中位数聚合 → 消除评分随机波动，标记高方差项
  5. 规则后修正 → 用客观数据校准主观评分

支持两种模式：
  - single: 逐条评测单个模型输出
  - pairwise: 两个模型输出配对对比
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from cn_eval.data_loader.schema import (
    DIMENSIONS,
    Prompt, ModelOutput, AlignedPair,
    EvalResult, PairwiseResult, DimensionScores,
)
from cn_eval.aligner.prompt_aligner import PromptAligner
from cn_eval.utils.llm_client import LLMClient
from cn_eval.utils.config import EvalConfig
from cn_eval.utils.text import (
    split_paragraphs, split_sentences, count_chars,
    ngram_repetition_rate, detect_template_endings, detect_assistant_phrases,
)
from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class UnifiedEvaluator(BaseEvaluator):
    """
    统一 LLM Judge 评测器。

    一个评测器覆盖 single + pairwise 两种模式，
    内置多轮一致性保障 + 规则预分析 + 位置反转消偏。
    """

    name = "unified"

    def __init__(self, config: EvalConfig, client: LLMClient):
        super().__init__(config)
        self.client = client
        self._system_prompt = self._load_system_prompt()

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("请使用 run_single() 或 run_pairwise()")

    # ──────────────────────────────────────────────────────────
    # 单模型评测
    # ──────────────────────────────────────────────────────────

    async def run_single(
        self,
        prompts: list[Prompt],
        model_outputs: list[ModelOutput],
    ) -> dict[str, Any]:
        prompt_map = {p.prompt_id: p for p in prompts}
        semaphore = asyncio.Semaphore(self.config.judge.concurrency)

        async def _eval(output: ModelOutput) -> EvalResult | None:
            prompt = prompt_map.get(output.prompt_id)
            if not prompt:
                return None
            async with semaphore:
                return await self._judge_single_item(prompt, output)

        tasks = [_eval(o) for o in model_outputs]
        raw = await asyncio.gather(*tasks)
        results = [r for r in raw if r is not None]
        summary = self._compute_single_summary(results)

        logger.info(
            "[UnifiedEval:single] 完成 %d 条, 平均分=%.2f",
            len(results), summary.get("mean_score", 0),
        )
        return {"results": results, "summary": summary}

    async def _judge_single_item(
        self, prompt: Prompt, output: ModelOutput,
    ) -> EvalResult | None:
        pre = self._pre_analyze(output.response)
        pre_text = self._format_pre_analysis(pre)
        user_msg = (
            f"## 问题\n{prompt.text}\n\n"
            f"## 模型回答\n{output.response}\n\n"
            f"{pre_text}"
        )

        rounds_data = await self._multi_round_call(user_msg)
        if not rounds_data:
            return None

        scores, consistency = self._aggregate_single_rounds(rounds_data)
        scores = self._apply_rule_corrections(scores, pre)
        reasoning = rounds_data[0].get("reasoning", "")
        raw_issues = rounds_data[0].get("key_issues", [])
        issues = raw_issues if isinstance(raw_issues, list) else []

        return EvalResult(
            prompt_id=output.prompt_id,
            model_version=output.model_version,
            scores=scores,
            pre_analysis=pre,
            judge_reasoning=reasoning,
            key_issues=issues,
            consistency=consistency,
        )

    # ──────────────────────────────────────────────────────────
    # 配对对比评测
    # ──────────────────────────────────────────────────────────

    async def run_pairwise(
        self,
        prompts: list[Prompt],
        baseline_outputs: list[ModelOutput],
        candidate_outputs: list[ModelOutput],
        **kwargs: Any,
    ) -> dict[str, Any]:
        aligner = PromptAligner(prompts)
        pairs = aligner.align_pair(
            baseline_outputs, candidate_outputs,
            baseline_version=self.config.baseline,
            candidate_version=kwargs.get("candidate_version", "candidate"),
        )
        if not pairs:
            logger.warning("[UnifiedEval:pairwise] 无可对齐数据")
            return {"results": [], "summary": {}}

        semaphore = asyncio.Semaphore(self.config.judge.concurrency)

        async def _eval(pair: AlignedPair) -> PairwiseResult | None:
            async with semaphore:
                return await self._judge_pairwise_item(pair)

        tasks = [_eval(p) for p in pairs]
        raw = await asyncio.gather(*tasks)
        results = [r for r in raw if r is not None]
        summary = self._compute_pairwise_summary(results, pairs)

        logger.info(
            "[UnifiedEval:pairwise] 完成 %d 对, A=%.0f%% B=%.0f%% tie=%.0f%%",
            len(results),
            summary.get("win_rate_a", 0) * 100,
            summary.get("win_rate_b", 0) * 100,
            summary.get("tie_rate", 0) * 100,
        )
        return {"results": results, "summary": summary, "pairs": pairs}

    async def _judge_pairwise_item(self, pair: AlignedPair) -> PairwiseResult | None:
        pre_a = self._pre_analyze(pair.baseline_response)
        pre_b = self._pre_analyze(pair.candidate_response)

        temps = self.config.consistency.temperatures
        num_rounds = self.config.consistency.num_rounds
        rounds_data: list[dict] = []

        for i in range(num_rounds):
            temp = temps[i % len(temps)]
            swap = (i % 2 == 1)

            if swap:
                pa_text = self._format_pre_analysis(pre_b, label="A")
                pb_text = self._format_pre_analysis(pre_a, label="B")
                user_msg = self._build_pairwise_prompt(
                    pair.prompt_text,
                    pair.candidate_response, pair.baseline_response,
                    pa_text, pb_text,
                )
            else:
                pa_text = self._format_pre_analysis(pre_a, label="A")
                pb_text = self._format_pre_analysis(pre_b, label="B")
                user_msg = self._build_pairwise_prompt(
                    pair.prompt_text,
                    pair.baseline_response, pair.candidate_response,
                    pa_text, pb_text,
                )

            try:
                result = await self.client.judge(
                    self._system_prompt, user_msg, temperature=temp,
                )
                result["_swapped"] = swap
                rounds_data.append(result)
            except Exception as e:
                logger.error(
                    "[UnifiedEval] pairwise round %d 失败 (prompt=%s): %s",
                    i, pair.prompt_id, e,
                )

        if not rounds_data:
            return None

        winner, scores_a, scores_b, consistency = self._aggregate_pairwise_rounds(rounds_data)
        scores_a = self._apply_rule_corrections(scores_a, pre_a)
        scores_b = self._apply_rule_corrections(scores_b, pre_b)
        reasoning = rounds_data[0].get("reasoning", "")

        return PairwiseResult(
            prompt_id=pair.prompt_id,
            model_a=pair.baseline_version,
            model_b=pair.candidate_version,
            winner=winner,
            judge_id="unified_multi_round",
            scores_a=scores_a,
            scores_b=scores_b,
            reasoning=reasoning,
            consistency=consistency,
        )

    # ──────────────────────────────────────────────────────────
    # 多轮 LLM 调用
    # ──────────────────────────────────────────────────────────

    async def _multi_round_call(self, user_msg: str) -> list[dict]:
        """对同一个 user_msg 执行多轮评审。"""
        temps = self.config.consistency.temperatures
        num_rounds = self.config.consistency.num_rounds
        rounds: list[dict] = []

        for i in range(num_rounds):
            temp = temps[i % len(temps)]
            try:
                result = await self.client.judge(
                    self._system_prompt, user_msg, temperature=temp,
                )
                rounds.append(result)
            except Exception as e:
                logger.error("[UnifiedEval] single round %d 失败: %s", i, e)

        return rounds

    # ──────────────────────────────────────────────────────────
    # 规则预分析
    # ──────────────────────────────────────────────────────────

    def _pre_analyze(self, text: str) -> dict[str, Any]:
        return {
            "repetition": self._analyze_repetition(text),
            "structure": self._analyze_structure(text),
            "style": self._analyze_style(text),
        }

    def _analyze_repetition(self, text: str) -> dict[str, Any]:
        ngram_rates: dict[str, float] = {}
        for n in self.config.long_answer.ngram_sizes:
            ngram_rates[f"char_{n}gram"] = round(
                ngram_repetition_rate(text, n=n, level="char"), 4,
            )
            ngram_rates[f"word_{n}gram"] = round(
                ngram_repetition_rate(text, n=n, level="word"), 4,
            )

        mid = len(text) // 2
        half_repeat = round(
            ngram_repetition_rate(text, n=4, level="char"), 4,
        ) if mid > 0 else 0.0

        return {
            "ngram_rates": ngram_rates,
            "half_split_repeat": half_repeat,
            "total_chars": count_chars(text),
        }

    def _analyze_structure(self, text: str) -> dict[str, Any]:
        paragraphs = split_paragraphs(text)
        sentences = split_sentences(text)
        total_chars = count_chars(text)

        para_lengths = [count_chars(p) for p in paragraphs]
        avg_para_len = sum(para_lengths) / max(len(para_lengths), 1)
        para_std = (
            sum((ln - avg_para_len) ** 2 for ln in para_lengths) / max(len(para_lengths), 1)
        ) ** 0.5

        heading_count = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))
        list_count = len(re.findall(r'^\s*[-*]\s', text, re.MULTILINE))
        numbered_count = len(re.findall(r'^\s*\d+[.、]\s', text, re.MULTILINE))

        return {
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "avg_paragraph_length": round(avg_para_len, 1),
            "paragraph_length_std": round(para_std, 1),
            "heading_count": heading_count,
            "list_item_count": list_count + numbered_count,
            "chars_per_sentence": round(total_chars / max(len(sentences), 1), 1),
        }

    def _analyze_style(self, text: str) -> dict[str, Any]:
        template_ends = detect_template_endings(
            text, self.config.long_answer.template_endings,
        )
        assistant_phrases = detect_assistant_phrases(text)
        return {
            "template_endings": template_ends,
            "template_ending_count": len(template_ends),
            "assistant_phrases": assistant_phrases,
            "assistant_phrase_count": len(assistant_phrases),
        }

    def _format_pre_analysis(self, pre: dict, label: str = "") -> str:
        """将预分析结果格式化为人类可读文本，嵌入 LLM prompt。"""
        rep = pre.get("repetition", {})
        struct = pre.get("structure", {})
        style = pre.get("style", {})

        tag = f"（回答 {label}）" if label else ""
        lines = [f"## 规则预分析报告{tag}"]

        ngram_rates = rep.get("ngram_rates", {})
        if ngram_rates:
            worst_key = max(ngram_rates, key=ngram_rates.get)
            worst_val = ngram_rates[worst_key]
            lines.append(f"- 总字数: {rep.get('total_chars', 0)}")
            lines.append(f"- 最高 n-gram 重复率: {worst_key} = {worst_val:.2%}")
            lines.append(f"- 前后半段重复率: {rep.get('half_split_repeat', 0):.2%}")

        lines.append(
            f"- 段落数 {struct.get('paragraph_count', 0)}, "
            f"句子数 {struct.get('sentence_count', 0)}, "
            f"平均段落长度 {struct.get('avg_paragraph_length', 0)} 字"
        )
        if struct.get("heading_count", 0):
            lines.append(f"- Markdown 标题 {struct['heading_count']} 个")
        if struct.get("list_item_count", 0):
            lines.append(f"- 列表项 {struct['list_item_count']} 个")

        if style.get("template_endings"):
            lines.append(f"- [警告] 模板化结尾: {style['template_endings']}")
        if style.get("assistant_phrases"):
            lines.append(f"- [警告] 助手化用语: {style['assistant_phrases']}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────
    # 多轮聚合
    # ──────────────────────────────────────────────────────────

    def _aggregate_single_rounds(
        self, rounds: list[dict],
    ) -> tuple[DimensionScores, dict]:
        all_scores: dict[str, list[float]] = {d: [] for d in DIMENSIONS}

        for r in rounds:
            sd = r.get("scores", {})
            for dim in DIMENSIONS:
                try:
                    all_scores[dim].append(float(sd.get(dim, 0)))
                except (TypeError, ValueError):
                    all_scores[dim].append(0.0)

        final, stds = {}, {}
        for dim in DIMENSIONS:
            vals = sorted(all_scores[dim])
            final[dim] = self._median(vals)
            stds[dim] = self._std(vals)

        max_std = max(stds.values()) if stds else 0
        consistency = {
            "num_rounds": len(rounds),
            "per_dimension_std": stds,
            "max_std": round(max_std, 3),
            "uncertain": max_std > self.config.consistency.variance_threshold,
        }

        scores = DimensionScores(**{d: round(final[d], 1) for d in DIMENSIONS})
        return scores, consistency

    def _aggregate_pairwise_rounds(
        self, rounds: list[dict],
    ) -> tuple[str, DimensionScores, DimensionScores, dict]:
        all_a: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
        all_b: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
        winners: list[str] = []

        for r in rounds:
            swapped = r.get("_swapped", False)
            w = r.get("winner", "tie")

            if swapped:
                w = {"A": "B", "B": "A"}.get(w, w)
                sa, sb = r.get("scores_b", {}), r.get("scores_a", {})
            else:
                sa, sb = r.get("scores_a", {}), r.get("scores_b", {})

            winners.append(w)
            for dim in DIMENSIONS:
                try:
                    all_a[dim].append(float(sa.get(dim, 0)))
                except (TypeError, ValueError):
                    all_a[dim].append(0.0)
                try:
                    all_b[dim].append(float(sb.get(dim, 0)))
                except (TypeError, ValueError):
                    all_b[dim].append(0.0)

        # 多数投票决定胜者
        vote = Counter(winners)
        mc = vote.most_common()
        if len(mc) == 1:
            final_winner = mc[0][0]
        elif mc[0][1] > mc[1][1]:
            final_winner = mc[0][0]
        else:
            final_winner = "tie"

        final_a = {d: round(self._median(all_a[d]), 1) for d in DIMENSIONS}
        final_b = {d: round(self._median(all_b[d]), 1) for d in DIMENSIONS}

        winner_agreement = mc[0][1] / len(winners) if winners else 0
        consistency = {
            "num_rounds": len(rounds),
            "winner_votes": dict(vote),
            "winner_agreement": round(winner_agreement, 3),
            "scores_a_std": {d: self._std(all_a[d]) for d in DIMENSIONS},
            "scores_b_std": {d: self._std(all_b[d]) for d in DIMENSIONS},
            "position_swaps": sum(1 for r in rounds if r.get("_swapped", False)),
        }

        return (
            final_winner,
            DimensionScores(**final_a),
            DimensionScores(**final_b),
            consistency,
        )

    # ──────────────────────────────────────────────────────────
    # 规则后修正
    # ──────────────────────────────────────────────────────────

    def _apply_rule_corrections(
        self, scores: DimensionScores, pre: dict,
    ) -> DimensionScores:
        adjusted = scores.model_copy()
        threshold = self.config.anomaly.ngram_repeat_threshold

        ngram_rates = pre.get("repetition", {}).get("ngram_rates", {})
        worst = max(ngram_rates.values()) if ngram_rates else 0
        if worst > threshold:
            penalty = min(2.0, (worst - threshold) * 5)
            adjusted.repetition = max(1.0, adjusted.repetition - penalty)

        style = pre.get("style", {})
        n_tpl = style.get("template_ending_count", 0)
        if n_tpl > 0:
            adjusted.naturalness = max(1.0, adjusted.naturalness - 0.5 * n_tpl)

        n_ast = style.get("assistant_phrase_count", 0)
        if n_ast > 2:
            adjusted.naturalness = max(1.0, adjusted.naturalness - 0.3 * min(n_ast, 5))

        return adjusted

    # ──────────────────────────────────────────────────────────
    # 统计汇总
    # ──────────────────────────────────────────────────────────

    def _compute_single_summary(self, results: list[EvalResult]) -> dict[str, Any]:
        if not results:
            return {}

        n = len(results)
        all_means = [r.scores.mean() for r in results]
        mean_score = sum(all_means) / n

        dim_avgs = {}
        for dim in DIMENSIONS:
            dim_avgs[dim] = sum(getattr(r.scores, dim) for r in results) / n

        low = sum(1 for m in all_means if m < 2.5)
        mid = sum(1 for m in all_means if 2.5 <= m < 3.5)
        high = sum(1 for m in all_means if m >= 3.5)

        uncertain_n = sum(1 for r in results if r.consistency.get("uncertain", False))
        avg_max_std = sum(r.consistency.get("max_std", 0) for r in results) / n

        thr = self.config.anomaly.ngram_repeat_threshold
        high_rep = sum(
            1 for r in results
            if max(r.pre_analysis.get("repetition", {}).get("ngram_rates", {}).values() or [0]) > thr
        )
        has_tpl = sum(
            1 for r in results
            if r.pre_analysis.get("style", {}).get("template_ending_count", 0) > 0
        )

        return {
            "total": n,
            "mean_score": round(mean_score, 3),
            "dimension_averages": {k: round(v, 3) for k, v in dim_avgs.items()},
            "quality_distribution": {
                "low (<2.5)": low, "mid (2.5-3.5)": mid, "high (>=3.5)": high,
            },
            "consistency": {
                "uncertain_count": uncertain_n,
                "uncertain_rate": round(uncertain_n / n, 3),
                "avg_max_std": round(avg_max_std, 3),
            },
            "rule_flags": {
                "high_repetition": high_rep,
                "template_endings": has_tpl,
            },
        }

    def _compute_pairwise_summary(
        self, results: list[PairwiseResult], pairs: list[AlignedPair],
    ) -> dict[str, Any]:
        total = len(results) or 1
        wins_a = sum(1 for r in results if r.winner == "A")
        wins_b = sum(1 for r in results if r.winner == "B")
        ties = sum(1 for r in results if r.winner == "tie")

        cat_stats: dict[str, dict] = {}
        pair_map = {p.prompt_id: p.category for p in pairs}
        for r in results:
            cat = pair_map.get(r.prompt_id, "unknown")
            if cat not in cat_stats:
                cat_stats[cat] = {"total": 0, "A": 0, "B": 0, "tie": 0}
            cat_stats[cat]["total"] += 1
            cat_stats[cat][r.winner if r.winner in ("A", "B") else "tie"] += 1

        dims_a = self._avg_dims([r.scores_a for r in results])
        dims_b = self._avg_dims([r.scores_b for r in results])

        avg_wa = sum(
            r.consistency.get("winner_agreement", 1) for r in results
        ) / total

        return {
            "total": total,
            "win_rate_a": wins_a / total,
            "win_rate_b": wins_b / total,
            "tie_rate": ties / total,
            "by_category": cat_stats,
            "avg_scores_a": dims_a,
            "avg_scores_b": dims_b,
            "consistency": {"avg_winner_agreement": round(avg_wa, 3)},
        }

    # ──────────────────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────────────────

    def _load_system_prompt(self) -> str:
        path = self.config.judge.prompt_template
        if path and Path(path).exists():
            return Path(path).read_text(encoding="utf-8")
        return (
            "你是一位专业的中文 SFT 数据质量评审专家。请根据多个维度对回答进行评分，"
            "以 JSON 格式输出结果。维度：format、structure、repetition、"
            "info_quality、naturalness、task_completion，每维度 1-5 整数。"
        )

    @staticmethod
    def _build_pairwise_prompt(
        prompt_text: str, resp_a: str, resp_b: str,
        pre_a_text: str, pre_b_text: str,
    ) -> str:
        return (
            f"## 问题\n{prompt_text}\n\n"
            f"## 回答 A\n{resp_a}\n\n"
            f"## 回答 B\n{resp_b}\n\n"
            f"{pre_a_text}\n\n"
            f"{pre_b_text}"
        )

    @staticmethod
    def _avg_dims(scores: list[DimensionScores]) -> dict[str, float]:
        if not scores:
            return {}
        n = len(scores)
        return {d: round(sum(getattr(s, d) for s in scores) / n, 3) for d in DIMENSIONS}

    @staticmethod
    def _median(vals: list[float]) -> float:
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2

    @staticmethod
    def _std(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return round((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5, 3)
