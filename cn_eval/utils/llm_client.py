"""
LLM API 调用封装 — 默认对接 MiniMax，兼容 OpenAI 格式 API。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

MINIMAX_DEFAULT_BASE_URL = "https://api.minimax.chat/v1"
MINIMAX_DEFAULT_MODEL = "MiniMax-Text-01"


class LLMClient:
    """统一 LLM 调用客户端，默认使用 MiniMax API。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = MINIMAX_DEFAULT_BASE_URL,
        model: str = MINIMAX_DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """发送聊天请求，返回文本响应。"""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        resp = await self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        logger.debug("[LLMClient] 响应长度: %d 字符", len(content))
        return content

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> dict:
        """发送请求并解析 JSON 响应。"""
        raw = await self.chat(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    async def judge(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> dict:
        """执行 Judge 调用，返回结构化 JSON。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.chat_json(messages, temperature=temperature)

    async def batch_judge(
        self,
        system_prompt: str,
        user_prompts: list[str],
        concurrency: int = 5,
        temperature: float | None = None,
    ) -> list[dict]:
        """并发批量 Judge。"""
        semaphore = asyncio.Semaphore(concurrency)
        results: list[dict | None] = [None] * len(user_prompts)

        async def _judge(idx: int, user_prompt: str) -> None:
            async with semaphore:
                try:
                    result = await self.judge(system_prompt, user_prompt, temperature)
                    results[idx] = result
                except Exception as e:
                    logger.error("[LLMClient] batch_judge 第 %d 条失败: %s", idx, e)
                    results[idx] = {"error": str(e)}

        await asyncio.gather(*[_judge(i, p) for i, p in enumerate(user_prompts)])
        return [r or {"error": "unknown"} for r in results]
