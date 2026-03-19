"""
A/B 还原器 — 将匿名化的 A/B 评测结果还原为真实模型名。
"""

from __future__ import annotations

import logging
from typing import Any

from cn_eval.data_loader.schema import ABMapping, PairwiseResult

logger = logging.getLogger(__name__)


class ABDecoder:
    """将 A/B 标识还原为真实模型版本名。"""

    def __init__(self, mappings: list[ABMapping]):
        self._map: dict[str, ABMapping] = {m.prompt_id: m for m in mappings}

    def decode_result(self, result: PairwiseResult) -> PairwiseResult:
        """将单条 PairwiseResult 中的 A/B 还原为真实模型名。"""
        mapping = self._map.get(result.prompt_id)
        if not mapping:
            logger.warning("prompt %s 无 A/B 映射，保持原样", result.prompt_id)
            return result

        decoded = result.model_copy()
        decoded.model_a = mapping.a_model
        decoded.model_b = mapping.b_model

        # 如果位置被交换过，winner 也要反转
        if result.position_swapped:
            if decoded.winner == "A":
                decoded.winner = "B"
            elif decoded.winner == "B":
                decoded.winner = "A"
            decoded.scores_a, decoded.scores_b = decoded.scores_b, decoded.scores_a

        return decoded

    def decode_batch(self, results: list[PairwiseResult]) -> list[PairwiseResult]:
        """批量还原。"""
        decoded = [self.decode_result(r) for r in results]
        logger.info("[ABDecoder] 还原 %d 条 A/B 结果", len(decoded))
        return decoded

    def get_model_pair(self, prompt_id: str) -> tuple[str, str] | None:
        """获取某 prompt 的真实模型对 (A模型, B模型)。"""
        m = self._map.get(prompt_id)
        if m:
            return (m.a_model, m.b_model)
        return None
