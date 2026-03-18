from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "4096")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))
    eval_temperature: float = field(default_factory=lambda: float(os.getenv("EVAL_TEMPERATURE", "0.3")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    concurrency: int = field(default_factory=lambda: int(os.getenv("CONCURRENCY", "5")))


@dataclass
class PipelineConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    input_path: Path = Path("input.jsonl")
    output_path: Path = Path("output.jsonl")
    report_path: Path = Path("report.json")
    mode: str = "generate"  # "generate" | "evaluate" | "cot"
    pass_threshold: float = 7.0  # 评分 >= 此值视为通过 (满分 10)
    max_rework_rounds: int = 2  # 打回重写最大轮次
    enable_dedup: bool = False          # 启用去重
    dedup_threshold: float = 0.85       # 去重相似度阈值
    enable_difficulty: bool = False     # 启用难度分级
    enable_augment: bool = False        # 启用数据增强
    augment_variants: int = 3           # 每题生成变体数
    enable_self_consistency: bool = False  # 启用多路采样验证
    consistency_samples: int = 5        # 多路采样数量
