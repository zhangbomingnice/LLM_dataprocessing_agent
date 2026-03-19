"""
Long-Answer Evaluator — 中文长回答专项评测。

在 LLM Judge 之外，叠加规则检测层：
  - n-gram 重复率分析
  - 前后半段一致性
  - 模板化结尾 / 助手化用语
  - 结构层次分析
"""

from __future__ import annotations

import logging
from typing import Any

from cn_eval.data_loader.schema import (
    Prompt, ModelOutput, LongAnswerResult, DimensionScores,
)
from cn_eval.judges.llm_judge import LLMJudge
from cn_eval.judges.rule_judge import RuleJudge
from cn_eval.utils.llm_client import LLMClient
from cn_eval.utils.config import EvalConfig
from cn_eval.utils.text import (
    split_paragraphs,
    split_sentences,
    ngram_repetition_rate,
    detect_template_endings,
    detect_assistant_phrases,
    count_chars,
)
from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class LongAnswerEvaluator(BaseEvaluator):
    """长回答专项评测器。"""

    name = "long_answer"

    def __init__(self, config: EvalConfig, client: LLMClient):
        super().__init__(config)
        self.client = client

    async def run(
        self,
        prompts: list[Prompt],
        model_outputs: list[ModelOutput],
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt_map = {p.prompt_id: p for p in prompts}
        results: list[LongAnswerResult] = []

        # LLM Judge
        llm_judge = LLMJudge(
            client=self.client,
            prompt_template_path=self.config.judge.prompt_template,
            judge_id="llm_long_answer",
            temperature=self.config.judge.temperature,
        )

        judge_items = [
            {
                "prompt_id": o.prompt_id,
                "prompt_text": prompt_map.get(o.prompt_id, Prompt(prompt_id="", text="")).text,
                "response_a": o.response,
            }
            for o in model_outputs
        ]
        llm_results = await llm_judge.judge_batch(
            judge_items, concurrency=self.config.judge.concurrency,
        )

        # 逐条整合 LLM 分数 + 规则分析
        for output, llm_raw in zip(model_outputs, llm_results):
            scores = LLMJudge.parse_scores(llm_raw)
            rep_stats = self._repetition_analysis(output.response)
            struct_stats = self._structure_analysis(output.response)
            style_stats = self._style_analysis(output.response)

            # 规则惩罚修正 LLM 分数
            adjusted = self._apply_rule_penalties(scores, rep_stats, style_stats)

            results.append(LongAnswerResult(
                prompt_id=output.prompt_id,
                model_version=output.model_version,
                scores=adjusted,
                repetition_stats=rep_stats,
                structure_stats=struct_stats,
                style_stats=style_stats,
                judge_reasoning=llm_raw.get("reasoning", ""),
            ))

        summary = self._compute_summary(results)

        logger.info(
            "[LongAnswerEval] 完成 %d 条, 平均分=%.2f",
            len(results), summary.get("mean_score", 0),
        )

        return {"results": results, "summary": summary}

    def _repetition_analysis(self, text: str) -> dict[str, Any]:
        """n-gram 重复率 + 前后半段对比。"""
        ngram_rates = {}
        for n in self.config.long_answer.ngram_sizes:
            ngram_rates[f"char_{n}gram"] = round(
                ngram_repetition_rate(text, n=n, level="char"), 4,
            )
            ngram_rates[f"word_{n}gram"] = round(
                ngram_repetition_rate(text, n=n, level="word"), 4,
            )

        # 前后半段重复对比
        mid = len(text) // 2
        first_half = text[:mid]
        second_half = text[mid:]
        half_repeat = round(
            ngram_repetition_rate(first_half + second_half, n=4, level="char"), 4,
        )

        return {
            "ngram_rates": ngram_rates,
            "half_split_repeat": half_repeat,
            "total_chars": count_chars(text),
        }

    def _structure_analysis(self, text: str) -> dict[str, Any]:
        """结构层次分析。"""
        paragraphs = split_paragraphs(text)
        sentences = split_sentences(text)
        total_chars = count_chars(text)

        # 段落长度方差
        para_lengths = [count_chars(p) for p in paragraphs]
        avg_para_len = sum(para_lengths) / max(len(para_lengths), 1)
        para_variance = (
            sum((l - avg_para_len) ** 2 for l in para_lengths) / max(len(para_lengths), 1)
        ) ** 0.5

        # Markdown 格式检测
        import re
        heading_count = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))
        list_count = len(re.findall(r'^\s*[-*]\s', text, re.MULTILINE))
        numbered_count = len(re.findall(r'^\s*\d+[.、]\s', text, re.MULTILINE))

        return {
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "avg_paragraph_length": round(avg_para_len, 1),
            "paragraph_length_std": round(para_variance, 1),
            "heading_count": heading_count,
            "list_item_count": list_count + numbered_count,
            "chars_per_sentence": round(total_chars / max(len(sentences), 1), 1),
        }

    def _style_analysis(self, text: str) -> dict[str, Any]:
        """风格分析 — 模板化结尾 + 助手化用语。"""
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

    def _apply_rule_penalties(
        self,
        scores: DimensionScores,
        rep_stats: dict,
        style_stats: dict,
    ) -> DimensionScores:
        """根据规则分析结果对 LLM 分数做惩罚修正。"""
        adjusted = scores.model_copy()

        # 重复率过高 → 扣 non_repetition
        ngram_rates = rep_stats.get("ngram_rates", {})
        worst_rate = max(ngram_rates.values()) if ngram_rates else 0
        if worst_rate > self.config.anomaly.ngram_repeat_threshold:
            penalty = min(2.0, (worst_rate - self.config.anomaly.ngram_repeat_threshold) * 5)
            adjusted.non_repetition = max(1.0, adjusted.non_repetition - penalty)

        # 模板化结尾 → 扣 mode
        n_template = style_stats.get("template_ending_count", 0)
        if n_template > 0:
            adjusted.mode = max(1.0, adjusted.mode - 0.5 * n_template)

        # 助手化用语 → 扣 mode
        n_assistant = style_stats.get("assistant_phrase_count", 0)
        if n_assistant > 2:
            adjusted.mode = max(1.0, adjusted.mode - 0.3 * min(n_assistant, 5))

        return adjusted

    def _compute_summary(self, results: list[LongAnswerResult]) -> dict[str, Any]:
        if not results:
            return {}

        n = len(results)
        all_means = [r.scores.mean() for r in results]
        mean_score = sum(all_means) / n

        # 各维度均值
        dim_avgs = {}
        for dim in ["mode", "structure", "organization", "fluency", "non_repetition", "task_fit"]:
            dim_avgs[dim] = sum(getattr(r.scores, dim) for r in results) / n

        # 问题分布
        low_count = sum(1 for m in all_means if m < 2.5)
        mid_count = sum(1 for m in all_means if 2.5 <= m < 3.5)
        high_count = sum(1 for m in all_means if m >= 3.5)

        # 规则检出
        high_repeat = sum(
            1 for r in results
            if max(r.repetition_stats.get("ngram_rates", {}).values() or [0])
               > self.config.anomaly.ngram_repeat_threshold
        )
        has_template = sum(
            1 for r in results if r.style_stats.get("template_ending_count", 0) > 0
        )

        return {
            "total": n,
            "mean_score": round(mean_score, 3),
            "dimension_averages": {k: round(v, 3) for k, v in dim_avgs.items()},
            "quality_distribution": {
                "low (<2.5)": low_count,
                "mid (2.5-3.5)": mid_count,
                "high (>=3.5)": high_count,
            },
            "rule_flags": {
                "high_repetition": high_repeat,
                "template_endings": has_template,
            },
        }
