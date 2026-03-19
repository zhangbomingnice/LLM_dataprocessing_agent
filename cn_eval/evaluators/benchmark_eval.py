"""
Benchmark Evaluator — 基准评测。

对接标准基准数据集，支持 exact-match / F1 / ROUGE 等客观指标。
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from cn_eval.data_loader.schema import Prompt, ModelOutput
from cn_eval.utils.config import EvalConfig
from cn_eval.utils.text import tokenize
from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class BenchmarkEvaluator(BaseEvaluator):
    """基准评测器 — 计算客观指标。"""

    name = "benchmark"

    def __init__(self, config: EvalConfig):
        super().__init__(config)

    async def run(
        self,
        prompts: list[Prompt],
        model_outputs: list[ModelOutput],
        reference_answers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        reference_answers: {prompt_id: reference_answer_text}
        """
        if not reference_answers:
            logger.warning("[BenchmarkEval] 无参考答案，跳过")
            return {"results": [], "summary": {}}

        output_map = {o.prompt_id: o for o in model_outputs}
        results = []

        for pid, ref in reference_answers.items():
            output = output_map.get(pid)
            if not output:
                continue

            pred = output.response.strip()
            ref_text = ref.strip()

            metrics = {
                "prompt_id": pid,
                "model_version": output.model_version,
                "exact_match": self._exact_match(pred, ref_text),
                "char_f1": self._char_f1(pred, ref_text),
                "token_f1": self._token_f1(pred, ref_text),
                "contains_answer": self._contains_answer(pred, ref_text),
                "rouge_l": self._rouge_l(pred, ref_text),
            }
            results.append(metrics)

        summary = self._compute_summary(results)
        logger.info(
            "[BenchmarkEval] 完成 %d 条, EM=%.1f%%, F1=%.3f",
            len(results),
            summary.get("exact_match", 0) * 100,
            summary.get("avg_token_f1", 0),
        )

        return {"results": results, "summary": summary}

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[。，！？、；：""''（）【】]', '', text)
        return text

    def _exact_match(self, pred: str, ref: str) -> bool:
        return self._normalize(pred) == self._normalize(ref)

    def _char_f1(self, pred: str, ref: str) -> float:
        pred_chars = list(self._normalize(pred))
        ref_chars = list(self._normalize(ref))
        return self._f1_score(pred_chars, ref_chars)

    def _token_f1(self, pred: str, ref: str) -> float:
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)
        pred_tokens = [t for t in pred_tokens if t.strip()]
        ref_tokens = [t for t in ref_tokens if t.strip()]
        return self._f1_score(pred_tokens, ref_tokens)

    def _contains_answer(self, pred: str, ref: str) -> bool:
        """检查预测是否包含参考答案。"""
        ref_norm = self._normalize(ref)
        pred_norm = self._normalize(pred)
        if len(ref_norm) < 5:
            return ref_norm in pred_norm
        # 允许参考答案 80% 的字在预测中
        ref_chars = set(ref_norm)
        match = sum(1 for c in ref_chars if c in pred_norm)
        return match / max(len(ref_chars), 1) > 0.8

    def _rouge_l(self, pred: str, ref: str) -> float:
        """ROUGE-L (基于 LCS)。"""
        pred_chars = list(self._normalize(pred))
        ref_chars = list(self._normalize(ref))

        if not pred_chars or not ref_chars:
            return 0.0

        lcs_len = self._lcs_length(pred_chars, ref_chars)
        precision = lcs_len / len(pred_chars)
        recall = lcs_len / len(ref_chars)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _lcs_length(a: list, b: list) -> int:
        m, n = len(a), len(b)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[n]

    @staticmethod
    def _f1_score(pred_tokens: list, ref_tokens: list) -> float:
        if not pred_tokens or not ref_tokens:
            return 0.0

        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        common = sum((pred_counter & ref_counter).values())

        if common == 0:
            return 0.0

        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def _compute_summary(self, results: list[dict]) -> dict[str, Any]:
        if not results:
            return {}

        n = len(results)
        em = sum(1 for r in results if r["exact_match"]) / n
        avg_char_f1 = sum(r["char_f1"] for r in results) / n
        avg_token_f1 = sum(r["token_f1"] for r in results) / n
        avg_rouge_l = sum(r["rouge_l"] for r in results) / n
        contains = sum(1 for r in results if r["contains_answer"]) / n

        return {
            "total": n,
            "exact_match": round(em, 4),
            "avg_char_f1": round(avg_char_f1, 4),
            "avg_token_f1": round(avg_token_f1, 4),
            "avg_rouge_l": round(avg_rouge_l, 4),
            "contains_answer_rate": round(contains, 4),
        }
