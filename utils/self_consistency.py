"""
Self-Consistency 多路采样投票。

核心思想（Wang et al., 2022）：
对同一道题生成 N 条不同推理路径，对最终答案做多数投票。
多路径收敛到同一答案 → 该答案大概率正确。
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from typing import Any

from agents.base import BaseAgent
from config import LLMConfig
from utils.schema import CorpusItem, CoTStep

logger = logging.getLogger(__name__)


class SelfConsistencyChecker:
    """对同一问题进行多路 CoT 采样，通过投票确定最可信的答案。"""

    def __init__(self, llm_config: LLMConfig, n_samples: int = 5):
        self.llm_config = llm_config
        self.n_samples = n_samples
        self._client: Any = None

    async def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
            )
        return self._client

    async def check(
        self,
        question: str,
        system_prompt: str = "请解答以下问题，给出分步推理过程和最终答案。最终答案请用 【答案】...【/答案】 包裹。",
    ) -> dict:
        """
        生成 N 条推理路径，返回投票结果。

        Returns:
            {
                "best_answer": 票数最多的答案,
                "confidence": 最高票数 / 总路径数,
                "vote_distribution": {"答案A": 3, "答案B": 2},
                "paths": [每条路径的完整推理],
                "is_consistent": 是否存在多数一致（confidence >= 0.6）
            }
        """
        client = await self._get_client()

        tasks = [
            self._generate_one_path(client, question, system_prompt, temp)
            for temp in self._sample_temperatures()
        ]
        paths = await asyncio.gather(*tasks, return_exceptions=True)

        valid_paths = []
        answers = []
        for p in paths:
            if isinstance(p, Exception):
                logger.warning("采样路径失败: %s", p)
                continue
            valid_paths.append(p)
            ans = self._extract_answer(p)
            answers.append(self._normalize_answer(ans))

        if not answers:
            return {
                "best_answer": "",
                "confidence": 0.0,
                "vote_distribution": {},
                "paths": [],
                "is_consistent": False,
            }

        counter = Counter(answers)
        best_answer, best_count = counter.most_common(1)[0]
        confidence = best_count / len(answers)

        return {
            "best_answer": best_answer,
            "confidence": round(confidence, 2),
            "vote_distribution": dict(counter),
            "paths": valid_paths,
            "is_consistent": confidence >= 0.6,
        }

    def _sample_temperatures(self) -> list[float]:
        """生成不同温度以产生多样化的推理路径。"""
        base = 0.7
        spread = 0.3
        temps = []
        for i in range(self.n_samples):
            t = base + (i - self.n_samples // 2) * (spread / self.n_samples)
            temps.append(max(0.1, min(1.2, round(t, 2))))
        return temps

    async def _generate_one_path(
        self,
        client: Any,
        question: str,
        system_prompt: str,
        temperature: float,
    ) -> str:
        resp = await client.chat.completions.create(
            model=self.llm_config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=temperature,
            max_tokens=self.llm_config.max_tokens,
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _extract_answer(text: str) -> str:
        """从推理文本中提取最终答案。"""
        patterns = [
            r'【答案】(.+?)【/答案】',
            r'最终答案[是为：:]\s*(.+?)(?:\n|$)',
            r'[Aa]nswer[：:]\s*(.+?)(?:\n|$)',
            r'因此[，,]\s*(.+?)(?:\n|$)',
            r'所以[，,]\s*(.+?)(?:\n|$)',
        ]
        for p in patterns:
            match = re.search(p, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        lines = text.strip().split('\n')
        return lines[-1].strip() if lines else ""

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """归一化答案以便比较（去除空白、标点差异等）。"""
        ans = answer.strip()
        ans = re.sub(r'[。，、；：！？\s]', '', ans)
        ans = re.sub(r'(\d)\s+', r'\1', ans)
        return ans.lower()
