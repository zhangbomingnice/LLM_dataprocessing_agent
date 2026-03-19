"""
EvalEngine — 评测主引擎，串联完整流程。

流程:
  1. 加载配置
  2. 加载数据（prompts + model outputs）
  3. 数据校验
  4. 按模式分发 → Evaluator
  5. 收集结果
  6. (后续第三/四期) 分析 → 报告
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from cn_eval.data_loader.loader import UnifiedLoader
from cn_eval.data_loader.schema import Prompt, ModelOutput, EvalMode
from cn_eval.data_loader.validators import DataValidator
from cn_eval.evaluators.pairwise_eval import PairwiseEvaluator
from cn_eval.evaluators.long_answer_eval import LongAnswerEvaluator
from cn_eval.evaluators.if_eval import IFEvaluator
from cn_eval.evaluators.benchmark_eval import BenchmarkEvaluator
from cn_eval.utils.config import EvalConfig, load_config
from cn_eval.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)
console = Console()


class EvalEngine:
    """评测主引擎。"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.client: LLMClient | None = None
        self.prompts: list[Prompt] = []
        self.outputs_by_version: dict[str, list[ModelOutput]] = {}
        self.results: dict[str, Any] = {}

    @classmethod
    def from_yaml(cls, config_path: str, overrides: dict | None = None) -> EvalEngine:
        config = load_config(config_path, overrides)
        return cls(config)

    def _init_client(self) -> LLMClient:
        if not self.config.judge.primary_api_key:
            raise ValueError("未设置 API Key，请在 .env 或配置文件中设置 MINIMAX_API_KEY")

        return LLMClient(
            api_key=self.config.judge.primary_api_key,
            base_url=self.config.judge.primary_api_base,
            model=self.config.judge.primary_model,
            temperature=self.config.judge.temperature,
            max_tokens=self.config.judge.max_tokens,
        )

    async def run(self) -> dict[str, Any]:
        """执行完整评测流程。"""
        console.rule("[bold cyan]中文长回答 SFT 评测系统")

        # 1. 初始化 LLM Client
        console.print("[dim]初始化 LLM Client...[/dim]")
        self.client = self._init_client()

        # 2. 加载数据
        console.print("[dim]加载数据...[/dim]")
        self._load_data()

        # 3. 校验
        console.print("[dim]校验数据完整性...[/dim]")
        self._validate_data()

        # 4. 按模式分发评测
        for mode_str in self.config.eval_modes:
            try:
                mode = EvalMode(mode_str)
            except ValueError:
                logger.warning("未知评测模式: %s，跳过", mode_str)
                continue

            console.rule(f"[bold yellow]评测模式: {mode.value}")

            if mode == EvalMode.PAIRWISE:
                result = await self._run_pairwise()
            elif mode == EvalMode.LONG_ANSWER:
                result = await self._run_long_answer()
            elif mode == EvalMode.IF_EVAL:
                result = await self._run_if_eval()
            elif mode == EvalMode.BENCHMARK:
                result = await self._run_benchmark()
            else:
                logger.warning("模式 %s 暂未实现", mode.value)
                continue

            self.results[mode.value] = result
            self._display_summary(mode.value, result.get("summary", {}))

        # 5. 保存结果
        self._save_results()

        console.rule("[bold green]评测完成")
        return self.results

    def _load_data(self) -> None:
        """加载测试题集和模型输出。"""
        if self.config.test_set_path:
            path = Path(self.config.test_set_path)
            if path.exists():
                self.prompts = UnifiedLoader.load_prompts(
                    path,
                    id_field=self.config.prompt_id_field,
                    text_field="text",
                    category_field=self.config.category_field,
                )
                console.print(f"  📋 加载 {len(self.prompts)} 条测试题")
            else:
                logger.warning("测试题集文件不存在: %s", path)

        for version, output_path in self.config.model_outputs.items():
            path = Path(output_path)
            if path.exists():
                outputs = UnifiedLoader.load_model_outputs(
                    path,
                    model_version=version,
                    id_field=self.config.prompt_id_field,
                    response_field=self.config.response_field,
                )
                self.outputs_by_version[version] = outputs
                console.print(f"  🤖 加载模型 [{version}] {len(outputs)} 条输出")
            else:
                logger.warning("模型输出文件不存在: %s (版本: %s)", path, version)

    def _validate_data(self) -> None:
        """数据校验。"""
        if self.prompts:
            warnings = DataValidator.validate_prompts(self.prompts)
            if warnings:
                for w in warnings:
                    console.print(f"  ⚠️  {w}", style="yellow")

        prompt_ids = {p.prompt_id for p in self.prompts}
        for version, outputs in self.outputs_by_version.items():
            warnings = DataValidator.validate_model_outputs(outputs, prompt_ids)
            if warnings:
                for w in warnings:
                    console.print(f"  ⚠️  [{version}] {w}", style="yellow")

    async def _run_pairwise(self) -> dict[str, Any]:
        """执行 Pairwise 评测。"""
        baseline = self.config.baseline
        candidates = self.config.candidates

        if not candidates:
            candidates = [v for v in self.outputs_by_version if v != baseline]

        if baseline not in self.outputs_by_version:
            console.print(f"  ❌ baseline [{baseline}] 无数据", style="red")
            return {"results": [], "summary": {}}

        all_results = {}
        evaluator = PairwiseEvaluator(self.config, self.client)

        for cand in candidates:
            if cand not in self.outputs_by_version:
                console.print(f"  ⚠️  candidate [{cand}] 无数据", style="yellow")
                continue

            console.print(f"  ⚔️  对比: [{baseline}] vs [{cand}]")
            result = await evaluator.run(
                prompts=self.prompts,
                baseline_outputs=self.outputs_by_version[baseline],
                candidate_outputs=self.outputs_by_version[cand],
                candidate_version=cand,
            )
            all_results[f"{baseline}_vs_{cand}"] = result

        return all_results if len(all_results) > 1 else next(iter(all_results.values()), {})

    async def _run_long_answer(self) -> dict[str, Any]:
        """执行长回答评测。"""
        evaluator = LongAnswerEvaluator(self.config, self.client)
        all_results = {}

        for version, outputs in self.outputs_by_version.items():
            console.print(f"  📝 评测模型 [{version}]")
            result = await evaluator.run(
                prompts=self.prompts,
                model_outputs=outputs,
            )
            all_results[version] = result

        return all_results

    async def _run_if_eval(self) -> dict[str, Any]:
        """执行指令遵循评测。"""
        evaluator = IFEvaluator(self.config, self.client)
        all_results = {}

        for version, outputs in self.outputs_by_version.items():
            console.print(f"  ✅ IF-Eval 模型 [{version}]")
            result = await evaluator.run(
                prompts=self.prompts,
                model_outputs=outputs,
            )
            all_results[version] = result

        return all_results

    async def _run_benchmark(self) -> dict[str, Any]:
        """执行基准评测。"""
        evaluator = BenchmarkEvaluator(self.config)

        ref_path = self.config.human_annotations_path
        reference_answers = {}
        if ref_path and Path(ref_path).exists():
            raw = UnifiedLoader.load_raw(Path(ref_path))
            for item in raw:
                pid = item.get(self.config.prompt_id_field, "")
                ref = item.get("reference", item.get("answer", ""))
                if pid and ref:
                    reference_answers[pid] = ref

        all_results = {}
        for version, outputs in self.outputs_by_version.items():
            console.print(f"  📊 Benchmark 模型 [{version}]")
            result = await evaluator.run(
                prompts=self.prompts,
                model_outputs=outputs,
                reference_answers=reference_answers,
            )
            all_results[version] = result

        return all_results

    def _display_summary(self, mode: str, summary: dict) -> None:
        """用 Rich 表格展示摘要。"""
        if not summary:
            return

        table = Table(title=f"{mode} 评测摘要", show_lines=True)
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")

        def _flatten(d: dict, prefix: str = "") -> list[tuple[str, str]]:
            rows = []
            for k, v in d.items():
                key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                if isinstance(v, dict):
                    rows.extend(_flatten(v, key))
                elif isinstance(v, float):
                    rows.append((key, f"{v:.4f}"))
                else:
                    rows.append((key, str(v)))
            return rows

        for key, val in _flatten(summary):
            table.add_row(key, val)

        console.print(table)

    def _save_results(self) -> None:
        """保存评测结果到文件。"""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results_path = out_dir / "eval_results.json"

        serializable = self._make_serializable(self.results)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        console.print(f"  💾 结果已保存: {results_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """将 Pydantic 模型等转换为可序列化 dict。"""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_serializable(i) for i in obj]
        return obj


async def run_eval(config_path: str, overrides: dict | None = None) -> dict[str, Any]:
    """便捷入口函数。"""
    engine = EvalEngine.from_yaml(config_path, overrides)
    return await engine.run()
