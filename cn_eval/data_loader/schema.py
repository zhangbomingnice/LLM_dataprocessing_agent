"""
评测系统核心数据模型。

六维度统一评分体系：
  format / structure / repetition / info_quality / naturalness / task_completion
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvalMode(str, Enum):
    PAIRWISE = "pairwise"
    SINGLE = "single"


# ── 输入数据模型 ──────────────────────────────────────────────

class Prompt(BaseModel):
    prompt_id: str
    text: str
    category: str = ""
    subset: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelOutput(BaseModel):
    prompt_id: str
    model_version: str
    response: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ABMapping(BaseModel):
    """单条 A/B 映射：prompt_id → {A: model_version, B: model_version}"""
    prompt_id: str
    a_model: str
    b_model: str


# ── 评测结果模型 ──────────────────────────────────────────────

DIMENSIONS = ["format", "structure", "repetition", "info_quality", "naturalness", "task_completion"]
DIM_LABELS_ZH = ["格式排版", "内容结构", "重复冗余", "信息质量", "语言自然度", "任务完成度"]


class DimensionScores(BaseModel):
    format: float = 0.0
    structure: float = 0.0
    repetition: float = 0.0
    info_quality: float = 0.0
    naturalness: float = 0.0
    task_completion: float = 0.0

    def mean(self) -> float:
        """六维度均值。"""
        vals = [self.format, self.structure, self.repetition,
                self.info_quality, self.naturalness, self.task_completion]
        return sum(vals) / len(vals)

    def to_dict(self) -> dict[str, float]:
        return {
            "format": self.format,
            "structure": self.structure,
            "repetition": self.repetition,
            "info_quality": self.info_quality,
            "naturalness": self.naturalness,
            "task_completion": self.task_completion,
        }


class EvalResult(BaseModel):
    """统一评测结果（单条回答）。"""
    prompt_id: str
    model_version: str
    scores: DimensionScores = Field(default_factory=DimensionScores)
    pre_analysis: dict[str, Any] = Field(default_factory=dict)
    judge_reasoning: str = ""
    key_issues: list[str] = Field(default_factory=list)
    consistency: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PairwiseResult(BaseModel):
    prompt_id: str
    model_a: str
    model_b: str
    winner: str = ""
    judge_id: str = ""
    scores_a: DimensionScores = Field(default_factory=DimensionScores)
    scores_b: DimensionScores = Field(default_factory=DimensionScores)
    reasoning: str = ""
    position_swapped: bool = False
    consistency: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── 对齐后的数据结构 ──────────────────────────────────────────

class AlignedPair(BaseModel):
    """按 prompt_id 对齐后的一对模型输出。"""
    prompt_id: str
    prompt_text: str
    category: str = ""
    baseline_version: str = ""
    baseline_response: str = ""
    candidate_version: str = ""
    candidate_response: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── 异常样本标记 ──────────────────────────────────────────────

class AnomalyFlag(BaseModel):
    prompt_id: str
    anomaly_types: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
