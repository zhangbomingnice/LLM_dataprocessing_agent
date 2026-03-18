from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import LLMConfig

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """所有 Agent 的基类，封装 LLM 调用和通用逻辑。"""

    name: str = "BaseAgent"

    def __init__(self, llm_config: LLMConfig, system_prompt: str = ""):
        self.llm_config = llm_config
        self.system_prompt = system_prompt
        self._client = AsyncOpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.llm_config.model,
            "messages": messages,
            "temperature": temperature or self.llm_config.temperature,
            "max_tokens": max_tokens or self.llm_config.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        resp = await self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        logger.debug("[%s] LLM response length: %d", self.name, len(content))
        return content

    async def _call_llm_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> dict:
        raw = await self._call_llm(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(raw)

    def _build_messages(self, user_content: str) -> list[dict[str, str]]:
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": user_content})
        return msgs

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        ...
