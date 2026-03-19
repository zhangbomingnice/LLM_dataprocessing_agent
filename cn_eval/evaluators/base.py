"""
Evaluator 基类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cn_eval.utils.config import EvalConfig


class BaseEvaluator(ABC):
    """所有评测器的抽象基类。"""

    name: str = "base"

    def __init__(self, config: EvalConfig):
        self.config = config

    @abstractmethod
    async def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        执行评测，返回结构化的评测结果。

        返回 dict 至少包含:
          - "results": list[...]  具体评测条目
          - "summary": dict       汇总统计
        """
