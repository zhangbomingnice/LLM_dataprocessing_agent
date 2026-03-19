"""
评测系统核心数据模型。

严格按蓝图定义：EvalMode / Prompt / ModelOutput / PairwiseResult / DimensionScores
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EvalMode(str, Enum):
    IF_EVAL = "if_eval"
    PAIRWISE = "pairwise"
    BENCHMARK = "benchmark"
    LONG_ANSWER = "long_answer"


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

class DimensionScores(BaseModel):
    mode: float = 0.0
    structure: float = 0.0
    organization: float = 0.0
    fluency: float = 0.0
    non_repetition: float = 0.0
    task_fit: float = 0.0
    factuality: Optional[float] = None
    info_density: Optional[float] = None
    executability: Optional[float] = None

    def mean(self) -> float:
        """核心六维度均值。"""
        core = [self.mode, self.structure, self.organization,
                self.fluency, self.non_repetition, self.task_fit]
        return sum(core) / len(core)

    def to_dict(self) -> dict[str, float]:
        d = {
            "mode": self.mode, "structure": self.structure,
            "organization": self.organization, "fluency": self.fluency,
            "non_repetition": self.non_repetition, "task_fit": self.task_fit,
        }
        if self.factuality is not None:
            d["factuality"] = self.factuality
        if self.info_density is not None:
            d["info_density"] = self.info_density
        if self.executability is not None:
            d["executability"] = self.executability
        return d


class PairwiseResult(BaseModel):
    prompt_id: str
    model_a: str
    model_b: str
    winner: str = ""  # "A" / "B" / "tie"
    judge_id: str = ""
    scores_a: DimensionScores = Field(default_factory=DimensionScores)
    scores_b: DimensionScores = Field(default_factory=DimensionScores)
    reasoning: str = ""
    position_swapped: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class IFResult(BaseModel):
    """Instruction-Following 硬验证结果。"""
    prompt_id: str
    model_version: str
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    details: str = ""


class LongAnswerResult(BaseModel):
    """长回答专项评测结果。"""
    prompt_id: str
    model_version: str
    scores: DimensionScores = Field(default_factory=DimensionScores)
    repetition_stats: dict[str, Any] = Field(default_factory=dict)
    structure_stats: dict[str, Any] = Field(default_factory=dict)
    style_stats: dict[str, Any] = Field(default_factory=dict)
    judge_reasoning: str = ""


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
