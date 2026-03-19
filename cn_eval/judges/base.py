"""
Judge 基类 — 定义所有 Judge 的公共接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseJudge(ABC):
    """所有 Judge 的抽象基类。"""

    judge_id: str = "base"

    @abstractmethod
    async def judge_single(
        self,
        prompt_text: str,
        response_a: str,
        response_b: str = "",
        **kwargs: Any,
    ) -> dict:
        """
        评审单条。

        Pairwise 模式：同时提供 response_a 和 response_b。
        单评模式：仅提供 response_a（response_b 为空）。

        返回 dict 包含 scores/winner/reasoning 等字段。
        """

    @abstractmethod
    async def judge_batch(
        self,
        items: list[dict],
        concurrency: int = 5,
        **kwargs: Any,
    ) -> list[dict]:
        """批量评审。"""
