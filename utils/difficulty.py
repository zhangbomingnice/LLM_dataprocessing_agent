"""
难度分级器 — 利用 LLM 和启发式规则对题目标注难度等级。

结合两种策略：
1. 启发式预估（基于文本特征：长度、关键词、公式密度等）
2. LLM 精细判断
"""

from __future__ import annotations

import logging
import re
from typing import Any

from utils.schema import CorpusItem

logger = logging.getLogger(__name__)

DIFFICULTY_KEYWORDS = {
    "hard": [
        "证明", "prove", "推导", "derive", "求证",
        "充分必要", "iff", "当且仅当",
        "广义", "generalize", "n维", "高阶",
        "微分方程", "偏导", "级数收敛", "特征值",
        "NP", "时间复杂度", "动态规划", "最优化",
    ],
    "medium": [
        "计算", "求解", "化简", "evaluate", "solve",
        "积分", "求导", "极限", "概率",
        "比较", "分析", "解释原因",
    ],
    "easy": [
        "什么是", "定义", "简述", "列举", "举例",
        "what is", "define", "list", "describe",
        "是否", "对错", "选择",
    ],
}


class DifficultyClassifier:
    """题目难度分级器。"""

    def classify_heuristic(self, item: CorpusItem) -> dict:
        """基于启发式规则进行难度预估。"""
        q = item.question.lower()
        score = 0
        signals: list[str] = []

        # 关键词匹配
        for kw in DIFFICULTY_KEYWORDS["hard"]:
            if kw.lower() in q:
                score += 3
                signals.append(f"硬关键词: {kw}")
        for kw in DIFFICULTY_KEYWORDS["medium"]:
            if kw.lower() in q:
                score += 1
                signals.append(f"中关键词: {kw}")
        for kw in DIFFICULTY_KEYWORDS["easy"]:
            if kw.lower() in q:
                score -= 1
                signals.append(f"易关键词: {kw}")

        # 公式密度
        formula_count = len(re.findall(r'\$[^$]+\$', item.question))
        formula_count += len(re.findall(r'\\\(.*?\\\)', item.question))
        if formula_count >= 3:
            score += 2
            signals.append(f"公式密度高: {formula_count}")
        elif formula_count >= 1:
            score += 1

        # 问题长度
        q_len = len(item.question)
        if q_len > 300:
            score += 2
            signals.append(f"题目较长: {q_len}字")
        elif q_len > 100:
            score += 1

        # 多问（包含多个问号或子问题）
        question_marks = item.question.count('？') + item.question.count('?')
        if question_marks >= 3:
            score += 2
            signals.append(f"多子问题: {question_marks}个")

        # 判定等级
        if score >= 5:
            level = "hard"
        elif score >= 2:
            level = "medium"
        else:
            level = "easy"

        return {
            "level": level,
            "score": score,
            "signals": signals,
        }

    def classify_batch(self, items: list[CorpusItem]) -> list[dict]:
        """批量分级。"""
        results = []
        for item in items:
            result = self.classify_heuristic(item)
            result["id"] = item.id
            results.append(result)

        stats = {"easy": 0, "medium": 0, "hard": 0}
        for r in results:
            stats[r["level"]] += 1

        logger.info(
            "[Difficulty] 分级完成: easy=%d, medium=%d, hard=%d",
            stats["easy"], stats["medium"], stats["hard"],
        )
        return results


DIFFICULTY_LLM_PROMPT = """\
请判断以下题目的难度等级。

# 难度标准
- **easy**: 基础概念题、定义题、简单计算（初中及以下）
- **medium**: 需要一定推理或多步计算（高中/大学低年级）
- **hard**: 需要复杂推导、证明、多概念综合（大学高年级/竞赛级）

# 输出格式
严格输出 JSON（不要包含 ```json 标记）：
{"level": "easy/medium/hard", "reason": "判断理由"}

# 题目
{question}
"""
