"""
数学验算引擎 — 用 SymPy 实际执行验证计算步骤。

不完全依赖 LLM 的"感觉"来判断计算是否正确，
而是把关键表达式丢给 SymPy 做符号计算验证。
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

try:
    import sympy
    from sympy import sympify, simplify, Eq, solve, integrate, diff, sqrt, oo, pi
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logger.warning("sympy 未安装，数学验算功能不可用。运行: pip install sympy")


class MathVerifier:
    """使用 SymPy 对数学表达式和计算步骤进行符号验证。"""

    def __init__(self):
        if not SYMPY_AVAILABLE:
            raise ImportError("需要 sympy 库: pip install sympy")

    def verify_equation(self, lhs: str, rhs: str) -> dict:
        """验证 lhs == rhs 是否成立。"""
        try:
            lhs_expr = self._parse(lhs)
            rhs_expr = self._parse(rhs)
            diff_expr = simplify(lhs_expr - rhs_expr)
            is_equal = diff_expr == 0
            return {
                "is_correct": is_equal,
                "lhs_parsed": str(lhs_expr),
                "rhs_parsed": str(rhs_expr),
                "difference": str(diff_expr),
                "error": None,
            }
        except Exception as e:
            return {
                "is_correct": None,
                "error": f"解析失败: {e}",
            }

    def verify_calculation(self, expression: str, expected_result: str) -> dict:
        """验证表达式的计算结果是否等于预期值。"""
        try:
            expr = self._parse(expression)
            expected = self._parse(expected_result)
            computed = simplify(expr)
            is_correct = simplify(computed - expected) == 0
            return {
                "is_correct": is_correct,
                "computed": str(computed),
                "expected": str(expected),
                "error": None,
            }
        except Exception as e:
            return {
                "is_correct": None,
                "error": f"计算验证失败: {e}",
            }

    def verify_integral(self, integrand: str, var: str, lower: str | None, upper: str | None, expected: str) -> dict:
        """验证积分计算是否正确。"""
        try:
            from sympy import Symbol, integrate
            x = Symbol(var)
            f = self._parse(integrand)

            if lower is not None and upper is not None:
                result = integrate(f, (x, self._parse(lower), self._parse(upper)))
            else:
                result = integrate(f, x)

            expected_expr = self._parse(expected)
            is_correct = simplify(result - expected_expr) == 0
            return {
                "is_correct": is_correct,
                "computed": str(result),
                "expected": str(expected_expr),
                "error": None,
            }
        except Exception as e:
            return {"is_correct": None, "error": str(e)}

    def verify_solve(self, equation: str, var: str, expected_solutions: list[str]) -> dict:
        """验证方程求解是否正确。"""
        try:
            from sympy import Symbol, solve, Eq
            x = Symbol(var)
            lhs, rhs = equation.split("=") if "=" in equation else (equation, "0")
            eq = Eq(self._parse(lhs), self._parse(rhs))
            solutions = solve(eq, x)
            solutions_str = sorted([str(s) for s in solutions])
            expected_str = sorted([str(self._parse(e)) for e in expected_solutions])
            is_correct = solutions_str == expected_str
            return {
                "is_correct": is_correct,
                "computed_solutions": solutions_str,
                "expected_solutions": expected_str,
                "error": None,
            }
        except Exception as e:
            return {"is_correct": None, "error": str(e)}

    def verify_step_formulas(self, steps: list[dict]) -> list[dict]:
        """
        批量验证 CoT 步骤中的公式。

        输入 steps 中每个 dict 应包含:
          - formula: 公式字符串（LaTeX 或 Python 表达式）
          - expected: 预期结果（可选）
          - verify_type: "equation" | "calculation" | "integral" | "solve"

        返回每步的验证结果列表。
        """
        results = []
        for step in steps:
            vtype = step.get("verify_type", "calculation")
            formula = step.get("formula", "")

            if not formula:
                results.append({"step": step.get("step_number"), "skipped": True})
                continue

            try:
                if vtype == "equation":
                    parts = re.split(r'\s*=\s*', formula, maxsplit=1)
                    if len(parts) == 2:
                        res = self.verify_equation(parts[0], parts[1])
                    else:
                        res = {"is_correct": None, "error": "无法拆分等式"}
                elif vtype == "calculation":
                    expected = step.get("expected", "")
                    res = self.verify_calculation(formula, expected) if expected else {"skipped": True}
                else:
                    res = {"skipped": True}
            except Exception as e:
                res = {"is_correct": None, "error": str(e)}

            res["step_number"] = step.get("step_number")
            results.append(res)

        return results

    def _parse(self, expr_str: str) -> Any:
        """尝试解析表达式，先试 LaTeX，失败再试 Python 语法。"""
        expr_str = expr_str.strip().strip('$')

        if SYMPY_AVAILABLE:
            try:
                return parse_latex(expr_str)
            except Exception:
                pass

        expr_str = expr_str.replace('^', '**').replace('{', '(').replace('}', ')')
        return sympify(expr_str)


def quick_verify(expression: str, expected: str) -> bool | None:
    """快速验证一个表达式是否等于预期值。不可用时返回 None。"""
    if not SYMPY_AVAILABLE:
        return None
    try:
        v = MathVerifier()
        result = v.verify_calculation(expression, expected)
        return result.get("is_correct")
    except Exception:
        return None
