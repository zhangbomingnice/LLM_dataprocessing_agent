from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    GENERATE = "generate"
    EVALUATE = "evaluate"
    COT = "cot"


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


# ── CoT 步骤结构 ──────────────────────────────────────────────
class CoTStep(BaseModel):
    step_number: int
    step_type: str = ""  # 如: 审题, 建模, 公式推导, 代入计算, 验证, 结论
    content: str
    formula: str = ""    # 该步骤涉及的核心公式（LaTeX）


class StepVerification(BaseModel):
    step_number: int
    is_correct: bool
    error_type: str = ""    # 如: 无误, 计算错误, 逻辑跳跃, 公式误用, 概念混淆
    explanation: str = ""
    suggested_fix: str = ""


class CoTEvalResult(BaseModel):
    item_id: str | int
    total_steps: int = 0
    correct_steps: int = 0
    first_error_step: int | None = None
    step_accuracy: float = 0.0
    overall_score: float = Field(ge=0, le=10, default=0)
    passed: bool = False
    step_verifications: list[StepVerification] = Field(default_factory=list)
    chain_coherence: float = Field(ge=0, le=10, default=0)  # 推理链连贯性
    final_answer_correct: bool = False
    suggestion: str = ""


# ── 最终输出条目 ──────────────────────────────────────────────
class OutputItem(BaseModel):
    id: str | int
    question: str
    answer: str
    cot_steps: list[CoTStep] | None = None
    evaluation: EvalResult | None = None
    cot_evaluation: CoTEvalResult | None = None
    rework_count: int = 0
