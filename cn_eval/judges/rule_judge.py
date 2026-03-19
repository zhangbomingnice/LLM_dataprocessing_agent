"""
Rule Judge — 基于规则的硬性验证评审员。

不依赖 LLM，使用确定性规则检查回答质量。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from cn_eval.utils.text import (
    ngram_repetition_rate,
    detect_template_endings,
    detect_assistant_phrases,
    split_sentences,
    count_chars,
)
from .base import BaseJudge

logger = logging.getLogger(__name__)


class RuleJudge(BaseJudge):
    """基于规则的评审员 — 覆盖重复率、模板化结尾、长度异常等硬指标。"""

    judge_id = "rule"

    def __init__(
        self,
        ngram_repeat_threshold: float = 0.3,
        min_length: int = 50,
        max_length: int = 50000,
        template_penalty: float = 0.5,
        assistant_phrase_penalty: float = 0.3,
    ):
        self.ngram_repeat_threshold = ngram_repeat_threshold
        self.min_length = min_length
        self.max_length = max_length
        self.template_penalty = template_penalty
        self.assistant_phrase_penalty = assistant_phrase_penalty

    async def judge_single(
        self,
        prompt_text: str,
        response_a: str,
        response_b: str = "",
        **kwargs: Any,
    ) -> dict:
        if response_b:
            checks_a = self._check_response(response_a)
            checks_b = self._check_response(response_b)
            score_a = checks_a.pop("overall_score")
            score_b = checks_b.pop("overall_score")

            if score_a > score_b + 0.2:
                winner = "A"
            elif score_b > score_a + 0.2:
                winner = "B"
            else:
                winner = "tie"

            return {
                "winner": winner,
                "judge_id": self.judge_id,
                "rule_scores_a": score_a,
                "rule_scores_b": score_b,
                "checks_a": checks_a,
                "checks_b": checks_b,
            }
        else:
            checks = self._check_response(response_a)
            return {
                "judge_id": self.judge_id,
                "rule_score": checks.pop("overall_score"),
                "checks": checks,
            }

    async def judge_batch(
        self,
        items: list[dict],
        concurrency: int = 5,
        **kwargs: Any,
    ) -> list[dict]:
        results = []
        for item in items:
            r = await self.judge_single(
                prompt_text=item.get("prompt_text", ""),
                response_a=item.get("response_a", ""),
                response_b=item.get("response_b", ""),
            )
            r["prompt_id"] = item.get("prompt_id", "")
            results.append(r)
        logger.info("[RuleJudge] 批量检查完成: %d 条", len(results))
        return results

    def _check_response(self, text: str) -> dict:
        """对单条回答执行所有规则检查。"""
        length = count_chars(text)
        sentences = split_sentences(text)

        ngram_repeat_3 = ngram_repetition_rate(text, n=3, level="char")
        ngram_repeat_4 = ngram_repetition_rate(text, n=4, level="char")
        template_ends = detect_template_endings(text)
        assistant_phrases = detect_assistant_phrases(text)

        # 长度检查
        length_ok = self.min_length <= length <= self.max_length

        # 重复率
        repeat_ok = ngram_repeat_4 < self.ngram_repeat_threshold

        score = 5.0
        penalties = []

        if not length_ok:
            score -= 1.5
            penalties.append(f"长度异常({length}字)")

        if not repeat_ok:
            penalty = min(2.0, (ngram_repeat_4 - self.ngram_repeat_threshold) * 5)
            score -= penalty
            penalties.append(f"n-gram重复率过高({ngram_repeat_4:.2f})")

        if template_ends:
            score -= self.template_penalty * len(template_ends)
            penalties.append(f"模板化结尾: {template_ends}")

        if assistant_phrases:
            score -= self.assistant_phrase_penalty * min(len(assistant_phrases), 3)
            penalties.append(f"助手化用语: {assistant_phrases[:3]}")

        # 句子数太少（过短回答）
        if len(sentences) < 3 and length > 100:
            score -= 0.5
            penalties.append("分句过少")

        score = max(1.0, min(5.0, score))

        return {
            "overall_score": round(score, 2),
            "length": length,
            "sentence_count": len(sentences),
            "ngram_repeat_3": round(ngram_repeat_3, 3),
            "ngram_repeat_4": round(ngram_repeat_4, 3),
            "length_ok": length_ok,
            "repeat_ok": repeat_ok,
            "template_endings": template_ends,
            "assistant_phrases": assistant_phrases,
            "penalties": penalties,
        }
