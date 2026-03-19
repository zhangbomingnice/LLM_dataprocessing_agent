"""
EvalEngine — 评测主引擎，串联完整流程。

流程:
  1. 加载配置 + 初始化 LLM Client
  2. 加载数据（prompts + model outputs）
  3. 数据校验
  4. 统一 LLM Judge 评测（single / pairwise）
  5. 分析（统计 + 异常检测 + 版本对比）
  6. 生成报告（Markdown + CSV + 图表）
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from cn_eval.data_loader.loader import UnifiedLoader
from cn_eval.data_loader.schema import (
    Prompt, ModelOutput, EvalMode, EvalResult, PairwiseResult,
)
from cn_eval.data_loader.validators import DataValidator
from cn_eval.evaluators.unified_eval import UnifiedEvaluator
from cn_eval.analyzers.basic_stats import StatsCalculator
from cn_eval.analyzers.anomaly_detector import AnomalyDetector
from cn_eval.analyzers.long_answer_analyzer import LongAnswerAnalyzer
from cn_eval.analyzers.version_compare import VersionComparer
from cn_eval.report.markdown_report import MarkdownReporter
from cn_eval.report.csv_report import CSVReporter
from cn_eval.report.chart_gen import ChartGenerator
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
        console.rule("[bold cyan]中文长回答 SFT 评测系统（统一 LLM Judge）")

        console.print("[dim]初始化 LLM Client...[/dim]")
        self.client = self._init_client()

        console.print("[dim]加载数据...[/dim]")
        self._load_data()

        console.print("[dim]校验数据完整性...[/dim]")
        self._validate_data()

        evaluator = UnifiedEvaluator(self.config, self.client)

        for mode_str in self.config.eval_modes:
            try:
                mode = EvalMode(mode_str)
            except ValueError:
                logger.warning("未知评测模式: %s，跳过", mode_str)
                continue

            console.rule(f"[bold yellow]评测模式: {mode.value}")

            if mode == EvalMode.PAIRWISE:
                result = await self._run_pairwise(evaluator)
            elif mode == EvalMode.SINGLE:
                result = await self._run_single(evaluator)
            else:
                continue

            self.results[mode.value] = result
            self._display_summary(mode.value, result.get("summary", {}))

        console.rule("[bold magenta]分析引擎")
        analysis = self._run_analysis()

        console.rule("[bold blue]生成报告")
        self._generate_reports(analysis)

        self._save_results()

        console.rule("[bold green]评测完成")
        return self.results

    def _load_data(self) -> None:
        if self.config.test_set_path:
            path = Path(self.config.test_set_path)
            if path.exists():
                self.prompts = UnifiedLoader.load_prompts(
                    path,
                    id_field=self.config.prompt_id_field,
                    text_field="text",
                    category_field=self.config.category_field,
                )
                console.print(f"  加载 {len(self.prompts)} 条测试题")

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
                console.print(f"  加载模型 [{version}] {len(outputs)} 条输出")

    def _validate_data(self) -> None:
        if self.prompts:
            warnings = DataValidator.validate_prompts(self.prompts)
            for w in warnings:
                console.print(f"  [yellow]{w}[/yellow]")

        prompt_ids = {p.prompt_id for p in self.prompts}
        for version, outputs in self.outputs_by_version.items():
            warnings = DataValidator.validate_model_outputs(outputs, prompt_ids)
            for w in warnings:
                console.print(f"  [yellow][{version}] {w}[/yellow]")

    async def _run_pairwise(self, evaluator: UnifiedEvaluator) -> dict[str, Any]:
        baseline = self.config.baseline
        candidates = self.config.candidates or [
            v for v in self.outputs_by_version if v != baseline
        ]

        if baseline not in self.outputs_by_version:
            console.print(f"  [red]baseline [{baseline}] 无数据[/red]")
            return {"results": [], "summary": {}}

        all_results = {}
        for cand in candidates:
            if cand not in self.outputs_by_version:
                continue
            console.print(f"  对比: [{baseline}] vs [{cand}]")
            result = await evaluator.run_pairwise(
                prompts=self.prompts,
                baseline_outputs=self.outputs_by_version[baseline],
                candidate_outputs=self.outputs_by_version[cand],
                candidate_version=cand,
            )
            all_results[f"{baseline}_vs_{cand}"] = result

        if len(all_results) == 1:
            return next(iter(all_results.values()))
        return all_results

    async def _run_single(self, evaluator: UnifiedEvaluator) -> dict[str, Any]:
        all_results = {}
        for version, outputs in self.outputs_by_version.items():
            console.print(f"  评测模型 [{version}]")
            result = await evaluator.run_single(
                prompts=self.prompts,
                model_outputs=outputs,
            )
            all_results[version] = result
        return all_results

    def _run_analysis(self) -> dict[str, Any]:
        analysis: dict[str, Any] = {}

        anomaly_detector = AnomalyDetector(self.config.anomaly)
        all_anomalies = []
        for version, outputs in self.outputs_by_version.items():
            flags = anomaly_detector.detect_from_outputs(outputs)
            if flags:
                all_anomalies.extend(flags)
                console.print(f"  [{version}] 检出 {len(flags)} 条异常")
        analysis["anomalies"] = all_anomalies

        # 单模型评测深度分析
        single_data = self.results.get("single", {})
        if single_data and isinstance(single_data, dict):
            la_analyzer = LongAnswerAnalyzer()
            la_analysis = {}
            for version, vdata in single_data.items():
                if isinstance(vdata, dict) and "results" in vdata:
                    results = vdata["results"]
                    if results and isinstance(results[0], EvalResult):
                        la_analysis[version] = la_analyzer.analyze_batch(results)
                        console.print(f"  [{version}] 深度分析完成")

                        la_flags = anomaly_detector.detect_from_eval_results(results)
                        all_anomalies.extend(la_flags)
            analysis["single_analysis"] = la_analysis

        # 版本对比
        comparer = VersionComparer()
        if single_data and len(single_data) >= 2:
            analysis["version_comparison"] = comparer.compare_single(single_data)
            console.print("  版本对比分析完成")

        pw_data = self.results.get("pairwise", {})
        if pw_data:
            if "results" in pw_data:
                analysis["pairwise_comparison"] = comparer.compare_pairwise(
                    {"default": pw_data},
                )
            elif any(isinstance(v, dict) and "results" in v for v in pw_data.values()):
                analysis["pairwise_comparison"] = comparer.compare_pairwise(pw_data)

        stats = StatsCalculator()
        for version, outputs in self.outputs_by_version.items():
            lengths = [len(o.response) for o in outputs]
            analysis.setdefault("length_stats", {})[version] = stats.basic(lengths)

        self.results["_analysis"] = analysis
        return analysis

    def _generate_reports(self, analysis: dict[str, Any]) -> None:
        out_dir = Path(self.config.output_dir)

        config_summary = {
            "项目": self.config.project_name,
            "评测模式": ", ".join(self.config.eval_modes),
            "Judge 模型": self.config.judge.primary_model,
            "一致性轮数": self.config.consistency.num_rounds,
            "模型版本": ", ".join(self.outputs_by_version.keys()),
            "测试题数": len(self.prompts),
        }

        if "markdown" in self.config.report_formats:
            reporter = MarkdownReporter()
            display = {k: v for k, v in self.results.items() if not k.startswith("_")}
            reporter.generate(display, config_summary, output_path=out_dir / "report.md")
            console.print("  Markdown 报告已生成")

        if "csv" in self.config.report_formats:
            csv_reporter = CSVReporter()
            csv_dir = out_dir / "csv"

            pw_data = self.results.get("pairwise", {})
            if pw_data:
                results = pw_data.get("results", [])
                if results and isinstance(results[0], PairwiseResult):
                    csv_reporter.export_pairwise(results, csv_dir / "pairwise.csv")

            single_data = self.results.get("single", {})
            for version, vdata in (single_data.items() if isinstance(single_data, dict) else []):
                if isinstance(vdata, dict) and "results" in vdata:
                    results = vdata["results"]
                    if results and isinstance(results[0], EvalResult):
                        csv_reporter.export_single(
                            results, csv_dir / f"single_{version}.csv",
                        )

            anomalies = analysis.get("anomalies", [])
            if anomalies:
                csv_reporter.export_anomalies(anomalies, csv_dir / "anomalies.csv")

            comparer = VersionComparer()
            if single_data and isinstance(single_data, dict):
                table_rows = comparer.summary_table(single_data)
                if table_rows:
                    csv_reporter.export_version_table(
                        table_rows, csv_dir / "version_compare.csv",
                    )
            console.print("  CSV 报告已生成")

        if self.config.report_charts:
            chart_gen = ChartGenerator(out_dir / "charts")

            single_data = self.results.get("single", {})
            version_scores = {}
            for version, vdata in (single_data.items() if isinstance(single_data, dict) else []):
                if isinstance(vdata, dict):
                    dim_avgs = vdata.get("summary", {}).get("dimension_averages", {})
                    if dim_avgs:
                        version_scores[version] = dim_avgs

            if version_scores:
                chart_gen.radar_chart(version_scores)
                chart_gen.bar_chart(version_scores)
                console.print("  图表已生成")

            pw_data = self.results.get("pairwise", {})
            pw_summary = pw_data.get("summary", {})
            if pw_summary:
                chart_gen.win_rate_pie(
                    pw_summary.get("win_rate_a", 0),
                    pw_summary.get("win_rate_b", 0),
                    pw_summary.get("tie_rate", 0),
                )

    def _display_summary(self, mode: str, summary: dict) -> None:
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
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "eval_results.json"
        serializable = self._make_serializable(self.results)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        console.print(f"  结果已保存: {results_path}")

    def _make_serializable(self, obj: Any) -> Any:
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
