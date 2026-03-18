"""
语料去重模块 — 基于文本相似度的近似去重。

支持两种策略：
1. 精确去重：完全相同的 question 去重
2. 近似去重：基于 N-gram + Jaccard 相似度，去除语义高度重复的条目
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from utils.schema import CorpusItem

logger = logging.getLogger(__name__)


class Deduplicator:
    """语料去重器。"""

    def __init__(self, similarity_threshold: float = 0.85, ngram_size: int = 3):
        self.similarity_threshold = similarity_threshold
        self.ngram_size = ngram_size

    def run(self, items: list[CorpusItem]) -> tuple[list[CorpusItem], list[CorpusItem]]:
        """
        执行去重。

        Returns:
            (保留的条目, 被去重的条目)
        """
        logger.info("[Dedup] 开始去重，共 %d 条...", len(items))

        # Phase 1: 精确去重
        exact_seen: dict[str, int] = {}
        phase1_keep: list[CorpusItem] = []
        phase1_removed: list[CorpusItem] = []

        for item in items:
            key = self._exact_key(item.question)
            if key in exact_seen:
                phase1_removed.append(item)
            else:
                exact_seen[key] = len(phase1_keep)
                phase1_keep.append(item)

        logger.info("[Dedup] 精确去重: %d → %d（去除 %d 条完全重复）",
                     len(items), len(phase1_keep), len(phase1_removed))

        # Phase 2: 近似去重（N-gram Jaccard）
        fingerprints: list[set[str]] = []
        phase2_keep: list[CorpusItem] = []
        phase2_removed: list[CorpusItem] = []

        for item in phase1_keep:
            ngrams = self._get_ngrams(item.question)
            is_dup = False

            for existing_fp in fingerprints:
                sim = self._jaccard(ngrams, existing_fp)
                if sim >= self.similarity_threshold:
                    is_dup = True
                    break

            if is_dup:
                phase2_removed.append(item)
            else:
                fingerprints.append(ngrams)
                phase2_keep.append(item)

        total_removed = phase1_removed + phase2_removed
        logger.info("[Dedup] 近似去重: %d → %d（去除 %d 条近似重复）",
                     len(phase1_keep), len(phase2_keep), len(phase2_removed))
        logger.info("[Dedup] 最终保留 %d 条，共去除 %d 条",
                     len(phase2_keep), len(total_removed))

        return phase2_keep, total_removed

    def _exact_key(self, text: str) -> str:
        """生成精确匹配的 key。"""
        normalized = re.sub(r'\s+', '', text.lower())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def _get_ngrams(self, text: str) -> set[str]:
        """提取字符级 N-gram。"""
        text = re.sub(r'\s+', '', text.lower())
        if len(text) < self.ngram_size:
            return {text}
        return {text[i:i+self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}

    @staticmethod
    def _jaccard(set_a: set[str], set_b: set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
