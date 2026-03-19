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
    secondary_models: list[str] = field(default_factory=list)
    secondary_sample_ratio: float = 0.2
    temperature: float = 0.0
    blind_ab: bool = True
    swap_ratio: float = 0.5
    prompt_template: str = ""
    aggregation: str = "majority_vote"
    concurrency: int = 5
    max_tokens: int = 4096


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
    similarity_model: str = "shibing624/text2vec-base-chinese"


@dataclass
class EvalConfig:
    """评测系统完整配置。"""
    project_name: str = "chinese-long-answer-eval"
    output_dir: str = "./outputs"

    # 数据路径
    test_set_path: str = ""
    model_outputs: dict[str, str] = field(default_factory=dict)
    answer_key_path: str = ""
    human_annotations_path: str = ""

    # 字段映射
    prompt_id_field: str = "prompt_id"
    response_field: str = "response"
    category_field: str = "category"

    # 评测模式
    eval_modes: list[str] = field(default_factory=lambda: ["pairwise", "long_answer"])
    baseline: str = "base"
    candidates: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=lambda: [
        "mode", "structure", "organization", "fluency", "non_repetition", "task_fit",
    ])

    # 子配置
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    long_answer: LongAnswerConfig = field(default_factory=LongAnswerConfig)

    # 报告
    report_formats: list[str] = field(default_factory=lambda: ["markdown", "csv"])
    report_charts: bool = True
    report_language: str = "zh"


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> EvalConfig:
    """
    从 YAML 文件加载配置，支持字段覆盖。

    overrides 会递归覆盖 YAML 中的同名字段。
    """
    path = Path(path)
    if not path.exists():
        logger.warning("配置文件不存在: %s，使用默认配置", path)
        return EvalConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = _parse_config(raw)

    if overrides:
        config = _apply_overrides(config, overrides)

    # 环境变量自动填充
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
    stats_sec = raw.get("stats", {})
    anomaly_sec = raw.get("anomaly", {})
    la_sec = raw.get("long_answer", {})
    report_sec = raw.get("report", {})

    judge_cfg = JudgeConfig(
        primary_model=judge_sec.get("primary", JudgeConfig.primary_model),
        primary_api_base=judge_sec.get("api_base", JudgeConfig.primary_api_base),
        primary_api_key=judge_sec.get("api_key", ""),
        secondary_models=judge_sec.get("secondary", []),
        secondary_sample_ratio=judge_sec.get("secondary_sample_ratio", 0.2),
        temperature=judge_sec.get("temperature", 0.0),
        blind_ab=judge_sec.get("blind_ab", True),
        swap_ratio=judge_sec.get("swap_ratio", 0.5),
        prompt_template=judge_sec.get("prompt_template", ""),
        aggregation=judge_sec.get("aggregation", "majority_vote"),
        concurrency=judge_sec.get("concurrency", 5),
        max_tokens=judge_sec.get("max_tokens", 4096),
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
        similarity_model=la_sec.get("similarity_model", _la_defaults.similarity_model),
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
        eval_modes=eval_sec.get("modes", ["pairwise", "long_answer"]),
        baseline=eval_sec.get("baseline", "base"),
        candidates=eval_sec.get("candidates", []),
        dimensions=eval_sec.get("dimensions", [
            "mode", "structure", "organization", "fluency", "non_repetition", "task_fit",
        ]),
        judge=judge_cfg,
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
