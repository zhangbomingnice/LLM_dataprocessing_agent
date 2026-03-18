from __future__ import annotations

import json
import logging
from typing import Any

from config import LLMConfig
from utils.schema import CorpusItem, PlannerOutput, TaskType, Domain, Constraint

from .base import BaseAgent

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
# Role
你是一名顶尖的 LLM 语料工程师与流程架构师。

# Task
分析用户提供的原始语料样本和需求描述，输出一份结构化的执行蓝图。

# 分析维度
1. **意图识别**：判断任务是 "generate"（生成/重写答案）还是 "evaluate"（评估打分）。
2. **领域判定**：从 [数学, 代码, 逻辑推理, 文学/创意写作, 通用百科, 其他] 中选一。
3. **约束提取**：提取语言、长度、逻辑深度、格式等硬性约束（3-5 条）。
4. **质量基准**：定义"完美回答"的具体标准。

# 输出要求
严格输出以下 JSON（不要包含 ```json 标记）：
{
  "task_type": "generate" 或 "evaluate",
  "domain": "数学/代码/逻辑推理/文学/创意写作/通用百科/其他",
  "constraints": [
    {"name": "约束名", "description": "详细描述"}
  ],
  "processor_system_prompt": "为执行 Agent 编写的专项 System Prompt（仅 generate 模式需要）",
  "evaluator_system_prompt": "为评分 Agent 编写的专项 System Prompt",
  "gold_standard": "对完美回答的详细定义",
  "few_shot_example": "一个 Few-shot 示例，包含 Question 和 Perfect Answer"
}
"""


class PlannerAgent(BaseAgent):
    """Agent 1: 分析任务需求，生成执行蓝图和下游 Agent 的动态 Prompt。"""

    name = "Planner"

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config, system_prompt=PLANNER_SYSTEM_PROMPT)

    async def run(
        self,
        user_instruction: str,
        samples: list[CorpusItem],
        mode_hint: str | None = None,
    ) -> PlannerOutput:
        sample_text = "\n".join(
            f"[样本 {i+1}]\nQ: {s.question}\nA: {s.answer}"
            for i, s in enumerate(samples[:5])
        )

        user_msg = f"""## 用户需求
{user_instruction}

## 语料样本（前 {min(len(samples), 5)} 条）
{sample_text}
"""
        if mode_hint:
            user_msg += f"\n## 模式提示\n用户指定模式: {mode_hint}\n"

        logger.info("[Planner] 正在分析任务需求...")
        result = await self._call_llm_json(
            self._build_messages(user_msg),
            temperature=self.llm_config.eval_temperature,
        )

        task_type_raw = result.get("task_type", "generate")
        task_type = TaskType.GENERATE if "gen" in task_type_raw.lower() else TaskType.EVALUATE

        domain_map = {d.value: d for d in Domain}
        domain_raw = result.get("domain", "其他")
        domain = domain_map.get(domain_raw, Domain.OTHER)

        constraints = [
            Constraint(name=c["name"], description=c["description"])
            for c in result.get("constraints", [])
        ]

        output = PlannerOutput(
            task_type=task_type,
            domain=domain,
            constraints=constraints,
            processor_system_prompt=result.get("processor_system_prompt", ""),
            evaluator_system_prompt=result.get("evaluator_system_prompt", ""),
            gold_standard=result.get("gold_standard", ""),
            few_shot_example=result.get("few_shot_example", ""),
        )

        logger.info(
            "[Planner] 分析完成 — 类型: %s, 领域: %s, 约束: %d 条",
            output.task_type.value,
            output.domain.value,
            len(output.constraints),
        )
        return output
