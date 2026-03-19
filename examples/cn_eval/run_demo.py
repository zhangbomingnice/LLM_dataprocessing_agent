"""
cn_eval 端到端 Demo

两种运行模式:
  1. dry-run (默认): 使用 Mock LLM，零 API 调用，验证全管线
  2. real: 连接 MiniMax API 真实评测

用法:
  # dry-run 验证管线
  python examples/cn_eval/run_demo.py

  # 真实 API 评测
  python examples/cn_eval/run_demo.py --real --api-key YOUR_KEY

  # 指定配置文件
  python examples/cn_eval/run_demo.py --real --config examples/cn_eval/eval_config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, PropertyMock

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.logging import RichHandler

console = Console()


# ──────────────────────────────────────────────────────────
# Mock LLM — 模拟 LLM 返回合理评分
# ──────────────────────────────────────────────────────────

MOCK_SINGLE_RESPONSES = {
    "cn_001": {
        "scores": {"format": 2, "structure": 2, "repetition": 3, "info_quality": 2, "naturalness": 2, "task_completion": 2},
        "reasoning": "回答过于简略，缺乏具体细节和深度分析。存在模板化结尾。",
        "key_issues": ["信息量不足", "缺乏具体案例", "模板化结尾"],
    },
    "cn_002": {
        "scores": {"format": 3, "structure": 3, "repetition": 1, "info_quality": 2, "naturalness": 2, "task_completion": 3},
        "reasoning": "存在明显的内容复读（应用列表重复出现了两次），且有助手化开头和结尾。",
        "key_issues": ["内容重复", "助手化用语", "深度不足"],
    },
    "cn_003": {
        "scores": {"format": 2, "structure": 3, "repetition": 3, "info_quality": 2, "naturalness": 2, "task_completion": 2},
        "reasoning": "列举了挑战和建议但过于简略，未达到题目要求的每段100字以上。",
        "key_issues": ["未满足字数要求", "论述不够深入", "模板化结尾"],
    },
    "cn_004": {
        "scores": {"format": 3, "structure": 3, "repetition": 4, "info_quality": 2, "naturalness": 3, "task_completion": 3},
        "reasoning": "满足了列举5点的要求，但每点缺乏展开说明和原文佐证。",
        "key_issues": ["缺乏展开论述", "无原文引用"],
    },
    "cn_005": {
        "scores": {"format": 2, "structure": 3, "repetition": 3, "info_quality": 2, "naturalness": 2, "task_completion": 2},
        "reasoning": "涵盖了光反应和暗反应的基本概念，但缺乏具体步骤细节。结尾模板化。",
        "key_issues": ["缺乏步骤细节", "模板化结尾", "内容空泛"],
    },
}

MOCK_SINGLE_RESPONSES_V2 = {
    "cn_001": {
        "scores": {"format": 5, "structure": 5, "repetition": 5, "info_quality": 5, "naturalness": 5, "task_completion": 5},
        "reasoning": "回答结构清晰，四大发明分段详述，包含具体历史细节和跨文明影响分析。语言自然流畅。",
        "key_issues": [],
    },
    "cn_002": {
        "scores": {"format": 5, "structure": 5, "repetition": 5, "info_quality": 5, "naturalness": 5, "task_completion": 5},
        "reasoning": "从叠加态、纠缠到NISQ阶段，解释准确深入。区分了适用与不适用场景，信息密度高。",
        "key_issues": [],
    },
    "cn_003": {
        "scores": {"format": 5, "structure": 5, "repetition": 5, "info_quality": 5, "naturalness": 5, "task_completion": 5},
        "reasoning": "数据详实（具体百分比、人数），每个挑战都给出了针对性解决方案，论述层次递进。",
        "key_issues": [],
    },
    "cn_004": {
        "scores": {"format": 5, "structure": 5, "repetition": 5, "info_quality": 5, "naturalness": 4, "task_completion": 5},
        "reasoning": "6个性格维度归纳准确，每点有原文支撑和深度解读，最后一段升华到文学意义层面。",
        "key_issues": [],
    },
    "cn_005": {
        "scores": {"format": 5, "structure": 5, "repetition": 5, "info_quality": 5, "naturalness": 5, "task_completion": 5},
        "reasoning": "从总反应方程到两个光系统和Calvin循环的详细步骤，专业且完整。化学式和数据准确。",
        "key_issues": [],
    },
}

MOCK_PAIRWISE_RESPONSES = {
    "cn_001": {
        "winner": "B",
        "scores_a": {"format": 2, "structure": 2, "repetition": 3, "info_quality": 2, "naturalness": 2, "task_completion": 2},
        "scores_b": {"format": 5, "structure": 5, "repetition": 5, "info_quality": 5, "naturalness": 5, "task_completion": 5},
        "reasoning": "B 远胜于 A：结构清晰、细节丰富、有跨文明传播的深度分析，A 只有概括性描述。",
    },
    "cn_002": {
        "winner": "B",
        "scores_a": {"format": 3, "structure": 3, "repetition": 1, "info_quality": 2, "naturalness": 2, "task_completion": 3},
        "scores_b": {"format": 5, "structure": 5, "repetition": 5, "info_quality": 5, "naturalness": 5, "task_completion": 5},
        "reasoning": "A 存在内容重复和助手化用语，B 解释清晰专业，区分了量子计算的优势和局限。",
    },
    "cn_003": {
        "winner": "B",
        "scores_a": {"format": 2, "structure": 3, "repetition": 3, "info_quality": 2, "naturalness": 2, "task_completion": 2},
        "scores_b": {"format": 5, "structure": 5, "repetition": 5, "info_quality": 5, "naturalness": 5, "task_completion": 5},
        "reasoning": "B 用具体数据支撑每个论点，A 过于概括且未达字数要求。",
    },
    "cn_004": {
        "winner": "B",
        "scores_a": {"format": 3, "structure": 3, "repetition": 4, "info_quality": 2, "naturalness": 3, "task_completion": 3},
        "scores_b": {"format": 5, "structure": 5, "repetition": 5, "info_quality": 5, "naturalness": 4, "task_completion": 5},
        "reasoning": "B 每个性格维度都有原文引证和深层解读，A 只列出了简单标签。",
    },
    "cn_005": {
        "winner": "B",
        "scores_a": {"format": 2, "structure": 3, "repetition": 3, "info_quality": 2, "naturalness": 2, "task_completion": 2},
        "scores_b": {"format": 5, "structure": 5, "repetition": 5, "info_quality": 5, "naturalness": 5, "task_completion": 5},
        "reasoning": "B 提供了完整的光合作用反应链路和化学方程，A 只有概括性描述。",
    },
}


def make_mock_response(prompt_id: str, is_pairwise: bool, swapped: bool = False) -> dict:
    """根据 prompt_id 生成模拟 LLM 响应，加入随机微扰模拟多轮波动。"""
    if is_pairwise:
        base = MOCK_PAIRWISE_RESPONSES.get(prompt_id, MOCK_PAIRWISE_RESPONSES["cn_001"])
        resp = json.loads(json.dumps(base))
        if swapped:
            resp["scores_a"], resp["scores_b"] = resp["scores_b"], resp["scores_a"]
            w = resp["winner"]
            resp["winner"] = {"A": "B", "B": "A"}.get(w, w)
        for key in ("scores_a", "scores_b"):
            for dim in resp[key]:
                jitter = random.choice([-1, 0, 0, 0, 1])
                resp[key][dim] = max(1, min(5, resp[key][dim] + jitter))
        return resp
    else:
        base = MOCK_SINGLE_RESPONSES.get(prompt_id, MOCK_SINGLE_RESPONSES["cn_001"])
        resp = json.loads(json.dumps(base))
        for dim in resp["scores"]:
            jitter = random.choice([-1, 0, 0, 0, 1])
            resp["scores"][dim] = max(1, min(5, resp["scores"][dim] + jitter))
        return resp


def make_mock_response_v2(prompt_id: str) -> dict:
    base = MOCK_SINGLE_RESPONSES_V2.get(prompt_id, MOCK_SINGLE_RESPONSES_V2["cn_001"])
    resp = json.loads(json.dumps(base))
    for dim in resp["scores"]:
        jitter = random.choice([-1, 0, 0, 0, 1])
        resp["scores"][dim] = max(1, min(5, resp["scores"][dim] + jitter))
    return resp


class MockLLMClient:
    """模拟 LLM Client，返回预设评分。"""

    def __init__(self):
        from cn_eval.utils.llm_client import TokenTracker
        self.tracker = TokenTracker()
        self.model = "mock-judge"
        self._call_count = 0

    async def judge(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> dict:
        self._call_count += 1

        # 模拟 token 用量
        class FakeUsage:
            prompt_tokens = len(system_prompt) // 2 + len(user_prompt) // 2
            completion_tokens = 200
            total_tokens = prompt_tokens + completion_tokens
        self.tracker.record(FakeUsage())

        # 从 prompt 中提取 prompt_id 信息
        prompt_id = "cn_001"
        for pid in ["cn_001", "cn_002", "cn_003", "cn_004", "cn_005"]:
            if pid in user_prompt:
                prompt_id = pid
                break

        is_pairwise = "## 回答 A" in user_prompt and "## 回答 B" in user_prompt

        if is_pairwise:
            swapped = False
            return make_mock_response(prompt_id, is_pairwise=True, swapped=swapped)
        else:
            response_section = user_prompt.split("## 模型回答")[-1] if "## 模型回答" in user_prompt else ""
            is_long_response = len(response_section) > 400
            if is_long_response:
                return make_mock_response_v2(prompt_id)
            return make_mock_response(prompt_id, is_pairwise=False)

    async def chat(self, messages, **kwargs):
        return json.dumps(await self.judge(messages[0]["content"], messages[-1]["content"]))

    async def chat_json(self, messages, **kwargs):
        return await self.judge(messages[0]["content"], messages[-1]["content"])


async def run_dry(config_path: str):
    """dry-run: Mock LLM 全管线验证。"""
    from cn_eval.engine import EvalEngine
    from cn_eval.utils.config import load_config

    config = load_config(config_path)
    config.judge.primary_api_key = "mock-key"
    config.consistency.num_rounds = 3
    config.consistency.temperatures = [0.0, 0.05, 0.1]
    config.output_dir = str(ROOT / "outputs" / "demo_dryrun")

    engine = EvalEngine(config)
    engine.client = MockLLMClient()

    console.rule("[bold cyan]DRY-RUN: Mock LLM 全管线验证")
    console.print("[dim]使用预设评分，不消耗 API 额度[/dim]\n")

    engine._load_data = engine._load_data.__func__.__get__(engine, type(engine))
    engine._validate_data = engine._validate_data.__func__.__get__(engine, type(engine))

    console.print("[dim]加载数据...[/dim]")
    engine._load_data()
    console.print("[dim]校验数据...[/dim]")
    engine._validate_data()

    from cn_eval.evaluators.unified_eval import UnifiedEvaluator
    from cn_eval.data_loader.schema import EvalMode
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn,
        BarColumn, TaskProgressColumn,
        TimeElapsedColumn, TimeRemainingColumn,
    )

    evaluator = UnifiedEvaluator(config, engine.client)

    for mode_str in config.eval_modes:
        try:
            mode = EvalMode(mode_str)
        except ValueError:
            continue

        console.rule(f"[bold yellow]评测模式: {mode.value}")

        if mode == EvalMode.PAIRWISE:
            baseline = config.baseline
            candidates = config.candidates or [
                v for v in engine.outputs_by_version if v != baseline
            ]
            for cand in candidates:
                if cand not in engine.outputs_by_version:
                    continue
                n_items = min(
                    len(engine.outputs_by_version[baseline]),
                    len(engine.outputs_by_version[cand]),
                )
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TextColumn("({task.completed}/{task.total})"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Pairwise [{baseline}] vs [{cand}]", total=n_items)
                    result = await evaluator.run_pairwise(
                        prompts=engine.prompts,
                        baseline_outputs=engine.outputs_by_version[baseline],
                        candidate_outputs=engine.outputs_by_version[cand],
                        candidate_version=cand,
                        on_progress=lambda: progress.advance(task),
                    )
                engine.results["pairwise"] = result

        elif mode == EvalMode.SINGLE:
            all_results = {}
            for version, outputs in engine.outputs_by_version.items():
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TextColumn("({task.completed}/{task.total})"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Single [{version}]", total=len(outputs))
                    result = await evaluator.run_single(
                        prompts=engine.prompts,
                        model_outputs=outputs,
                        on_progress=lambda: progress.advance(task),
                    )
                all_results[version] = result
            engine.results["single"] = all_results

        snap = engine.client.tracker.snapshot()
        console.print(f"  [dim]Token 累计: {snap['total_tokens']:,} ({snap['call_count']} 次调用)[/dim]")

        summary = engine.results.get(mode.value, {}).get("summary", {})
        if summary:
            engine._display_summary(mode.value, summary)

    console.rule("[bold magenta]分析引擎")
    analysis = engine._run_analysis()

    console.rule("[bold blue]生成报告")
    engine._generate_reports(analysis)

    engine._save_results()

    # Token 用量
    snap = engine.client.tracker.snapshot()
    from rich.table import Table
    table = Table(title="Token 用量统计 (Mock)", show_lines=True)
    table.add_column("指标", style="cyan")
    table.add_column("数值", style="green", justify="right")
    table.add_row("API 调用次数", f"{snap['call_count']:,}")
    table.add_row("Prompt Tokens", f"{snap['prompt_tokens']:,}")
    table.add_row("Completion Tokens", f"{snap['completion_tokens']:,}")
    table.add_row("Total Tokens", f"{snap['total_tokens']:,}")
    console.print(table)

    console.rule("[bold green]DRY-RUN 完成")
    out_dir = Path(config.output_dir)
    console.print(f"\n输出目录: {out_dir.resolve()}")
    console.print("生成文件:")
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            console.print(f"  {f.relative_to(out_dir)} ({size:,} bytes)")


async def run_real(config_path: str, api_key: str | None = None):
    """real: 使用真实 MiniMax API 评测。"""
    from cn_eval.engine import EvalEngine
    from cn_eval.utils.config import load_config

    config = load_config(config_path)
    if api_key:
        config.judge.primary_api_key = api_key

    if not config.judge.primary_api_key:
        console.print("[bold red]错误: 未提供 API Key[/bold red]")
        console.print("设置方法 (任选其一):")
        console.print("  1. --api-key YOUR_KEY")
        console.print("  2. 环境变量 MINIMAX_API_KEY=YOUR_KEY")
        console.print("  3. .env 文件中 MINIMAX_API_KEY=YOUR_KEY")
        sys.exit(1)

    engine = EvalEngine(config)
    results = await engine.run()
    return results


def main():
    parser = argparse.ArgumentParser(description="cn_eval 端到端 Demo")
    parser.add_argument(
        "--real", action="store_true",
        help="使用真实 API（默认 dry-run 模式）",
    )
    parser.add_argument(
        "--config", "-c", type=str,
        default="examples/cn_eval/eval_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument("--api-key", type=str, help="MiniMax API Key")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )

    os.chdir(str(ROOT))

    if args.real:
        asyncio.run(run_real(args.config, args.api_key))
    else:
        asyncio.run(run_dry(args.config))


if __name__ == "__main__":
    main()
