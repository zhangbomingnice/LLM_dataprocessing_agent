"""
配置系统 — YAML 加载 + 环境变量 + CLI 覆盖。
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class JudgeConfig:
    primary_model: str = "MiniMax-Text-01"
    primary_api_key: str = field(default_factory=lambda: os.getenv("MINIMAX_API_KEY", ""))
    primary_api_base: str = "https://api.minimax.chat/v1"
    temperature: float = 0.0
    prompt_template: str = ""
    concurrency: int = 5
    max_tokens: int = 4096
    max_response_chars: int = 6000


@dataclass
class ConsistencyConfig:
    """多轮评审一致性配置。"""
    num_rounds: int = 3
    temperatures: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1])
    variance_threshold: float = 1.5
    aggregation: str = "median"


@dataclass
class StatsConfig:
    trimmed_ratio: float = 0.1
    bootstrap_n: int = 1000
    confidence_level: float = 0.95
    hypothesis_tests: list[str] = field(default_factory=lambda: ["wilcoxon", "sign_test"])


@dataclass
class AnomalyConfig:
    common_anomaly_threshold: float = 2.0
    length_percentile: list[int] = field(default_factory=lambda: [5, 95])
    ngram_repeat_threshold: float = 0.3
    judge_disagreement_threshold: int = 2


@dataclass
class LongAnswerConfig:
    ngram_sizes: list[int] = field(default_factory=lambda: [3, 4, 5])
    template_endings: list[str] = field(default_factory=lambda: [
        "总之", "综上所述", "希望以上内容对你有所帮助", "以上就是",
    ])
    half_split_ratio: float = 0.5


@dataclass
class EvalConfig:
    """评测系统完整配置。"""
    project_name: str = "chinese-long-answer-eval"
    output_dir: str = "./outputs"

    test_set_path: str = ""
    model_outputs: dict[str, str] = field(default_factory=dict)
    answer_key_path: str = ""
    human_annotations_path: str = ""

    prompt_id_field: str = "prompt_id"
    response_field: str = "response"
    category_field: str = "category"

    eval_modes: list[str] = field(default_factory=lambda: ["pairwise", "single"])
    baseline: str = "base"
    candidates: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=lambda: [
        "format", "structure", "repetition", "info_quality", "naturalness", "task_completion",
    ])

    judge: JudgeConfig = field(default_factory=JudgeConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    long_answer: LongAnswerConfig = field(default_factory=LongAnswerConfig)

    report_formats: list[str] = field(default_factory=lambda: ["markdown", "csv"])
    report_charts: bool = True
    report_language: str = "zh"


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> EvalConfig:
    """从 YAML 文件加载配置。"""
    path = Path(path)
    if not path.exists():
        logger.warning("配置文件不存在: %s，使用默认配置", path)
        return EvalConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = _parse_config(raw)

    if overrides:
        config = _apply_overrides(config, overrides)

    if not config.judge.primary_api_key:
        config.judge.primary_api_key = os.getenv("MINIMAX_API_KEY", "")
    if not config.judge.primary_api_key:
        config.judge.primary_api_key = os.getenv("OPENAI_API_KEY", "")

    logger.info("[Config] 已加载配置: %s (模式: %s)", path, config.eval_modes)
    return config


def _parse_config(raw: dict) -> EvalConfig:
    """将嵌套 YAML dict 解析为 EvalConfig。"""
    project = raw.get("project", {})
    data = raw.get("data", {})
    eval_sec = raw.get("eval", {})
    judge_sec = raw.get("judge", {})
    consistency_sec = raw.get("consistency", {})
    stats_sec = raw.get("stats", {})
    anomaly_sec = raw.get("anomaly", {})
    la_sec = raw.get("long_answer", {})
    report_sec = raw.get("report", {})

    judge_cfg = JudgeConfig(
        primary_model=judge_sec.get("primary", JudgeConfig.primary_model),
        primary_api_base=judge_sec.get("api_base", JudgeConfig.primary_api_base),
        primary_api_key=judge_sec.get("api_key", ""),
        temperature=judge_sec.get("temperature", 0.0),
        prompt_template=judge_sec.get("prompt_template", ""),
        concurrency=judge_sec.get("concurrency", 5),
        max_tokens=judge_sec.get("max_tokens", 4096),
        max_response_chars=judge_sec.get("max_response_chars", 6000),
    )

    _consistency_defaults = ConsistencyConfig()
    consistency_cfg = ConsistencyConfig(
        num_rounds=consistency_sec.get("num_rounds", _consistency_defaults.num_rounds),
        temperatures=consistency_sec.get("temperatures", _consistency_defaults.temperatures),
        variance_threshold=consistency_sec.get("variance_threshold", _consistency_defaults.variance_threshold),
        aggregation=consistency_sec.get("aggregation", _consistency_defaults.aggregation),
    )

    stats_cfg = StatsConfig(
        trimmed_ratio=stats_sec.get("trimmed_ratio", 0.1),
        bootstrap_n=stats_sec.get("bootstrap_n", 1000),
        confidence_level=stats_sec.get("confidence_level", 0.95),
        hypothesis_tests=stats_sec.get("hypothesis_tests", ["wilcoxon", "sign_test"]),
    )

    anomaly_cfg = AnomalyConfig(
        common_anomaly_threshold=anomaly_sec.get("common_anomaly_threshold", 2.0),
        length_percentile=anomaly_sec.get("length_percentile", [5, 95]),
        ngram_repeat_threshold=anomaly_sec.get("ngram_repeat_threshold", 0.3),
        judge_disagreement_threshold=anomaly_sec.get("judge_disagreement_threshold", 2),
    )

    _la_defaults = LongAnswerConfig()
    la_cfg = LongAnswerConfig(
        ngram_sizes=la_sec.get("ngram_sizes", [3, 4, 5]),
        template_endings=la_sec.get("template_endings", _la_defaults.template_endings),
        half_split_ratio=la_sec.get("half_split_ratio", 0.5),
    )

    return EvalConfig(
        project_name=project.get("name", "chinese-long-answer-eval"),
        output_dir=project.get("output_dir", "./outputs"),
        test_set_path=data.get("test_set", ""),
        model_outputs=data.get("model_outputs", {}),
        answer_key_path=data.get("answer_key", ""),
        human_annotations_path=data.get("human_annotations", ""),
        prompt_id_field=data.get("prompt_id_field", "prompt_id"),
        response_field=data.get("response_field", "response"),
        category_field=data.get("category_field", "category"),
        eval_modes=eval_sec.get("modes", ["pairwise", "single"]),
        baseline=eval_sec.get("baseline", "base"),
        candidates=eval_sec.get("candidates", []),
        dimensions=eval_sec.get("dimensions", [
            "format", "structure", "repetition", "info_quality", "naturalness", "task_completion",
        ]),
        judge=judge_cfg,
        consistency=consistency_cfg,
        stats=stats_cfg,
        anomaly=anomaly_cfg,
        long_answer=la_cfg,
        report_formats=report_sec.get("formats", ["markdown", "csv"]),
        report_charts=report_sec.get("charts", True),
        report_language=report_sec.get("language", "zh"),
    )


def _apply_overrides(config: EvalConfig, overrides: dict[str, Any]) -> EvalConfig:
    """将 CLI 覆盖参数应用到配置。"""
    for key, val in overrides.items():
        if val is None:
            continue
        if hasattr(config, key):
            setattr(config, key, val)
        elif "." in key:
            parts = key.split(".", 1)
            sub = getattr(config, parts[0], None)
            if sub and hasattr(sub, parts[1]):
                setattr(sub, parts[1], val)
    return config
