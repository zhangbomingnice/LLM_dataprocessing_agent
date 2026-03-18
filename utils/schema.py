from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    GENERATE = "generate"
    EVALUATE = "evaluate"


class Domain(str, Enum):
    MATH = "数学"
    CODE = "代码"
    LOGIC = "逻辑推理"
    LITERATURE = "文学/创意写作"
    ENCYCLOPEDIA = "通用百科"
    OTHER = "其他"


# ── 原始语料条目 ──────────────────────────────────────────────
class CorpusItem(BaseModel):
    id: str | int = ""
    question: str
    answer: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Planner 输出 ─────────────────────────────────────────────
class Constraint(BaseModel):
    name: str
    description: str


class PlannerOutput(BaseModel):
    task_type: TaskType
    domain: Domain
    constraints: list[Constraint]
    processor_system_prompt: str = ""
    evaluator_system_prompt: str = ""
    gold_standard: str = ""
    few_shot_example: str = ""


# ── Evaluator 输出 ────────────────────────────────────────────
class DimensionScore(BaseModel):
    dimension: str
    score: float = Field(ge=0, le=10)
    reason: str


class EvalResult(BaseModel):
    item_id: str | int
    total_score: float = Field(ge=0, le=10)
    passed: bool = False
    dimensions: list[DimensionScore] = Field(default_factory=list)
    suggestion: str = ""


# ── 最终输出条目 ──────────────────────────────────────────────
class OutputItem(BaseModel):
    id: str | int
    question: str
    answer: str
    evaluation: EvalResult | None = None
    rework_count: int = 0
