from __future__ import annotations

import asyncio
import logging
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from config import PipelineConfig
from utils.schema import (
    CorpusItem, OutputItem, PlannerOutput, EvalResult, TaskType,
)
from utils.io import read_jsonl, write_jsonl
from utils.file_parser import load_file, build_extraction_prompt, parse_extracted_json
from utils.file_writer import write_output
from agents.planner import PlannerAgent
from agents.processor import ProcessorAgent
from agents.evaluator import EvaluatorAgent
from agents.aggregator import AggregatorAgent

logger = logging.getLogger(__name__)
console = Console()


class Pipeline:
    """四阶段 Agent 流水线编排引擎。"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    async def run(self, user_instruction: str) -> dict:
        """执行完整流水线，返回汇总报告。"""
        console.print(Panel("[bold cyan]语料工程 Agent 流水线启动[/bold cyan]", expand=False))

        # ── Stage 0: 加载数据（自动识别格式） ─────────────────
        console.print("\n[bold]▶ Stage 0:[/bold] 加载语料文件...")
        items = await self._load_input(user_instruction)
        console.print(f"  已加载 [green]{len(items)}[/green] 条数据 ← {self.config.input_path}")

        # ── Stage 1: Planner ───────────────────────────────────
        console.print("\n[bold]▶ Stage 1:[/bold] Planner 分析任务需求...")
        planner = PlannerAgent(self.config.llm)
        plan = await planner.run(user_instruction, items, mode_hint=self.config.mode)
        self._display_plan(plan)

        # ── 根据模式分流 ───────────────────────────────────────
        if plan.task_type == TaskType.GENERATE:
            return await self._run_generate_pipeline(items, plan)
        else:
            return await self._run_evaluate_pipeline(items, plan)

    # ── 生成模式流水线 ─────────────────────────────────────────
    async def _run_generate_pipeline(
        self,
        items: list[CorpusItem],
        plan: PlannerOutput,
    ) -> dict:
        cfg = self.config

        # ── Stage 2: Processor 批量生成 ────────────────────────
        console.print("\n[bold]▶ Stage 2:[/bold] Processor 生成回答...")
        processor = ProcessorAgent(cfg.llm, plan=plan)
        answers = await self._run_with_progress(
            processor, items, "生成中"
        )

        output_items = [
            OutputItem(id=item.id, question=item.question, answer=answers.get(item.id, ""))
            for item in items
        ]

        # ── Stage 3: Evaluator 评审 + 打回重写循环 ─────────────
        console.print("\n[bold]▶ Stage 3:[/bold] Evaluator 质量评审...")
        evaluator = EvaluatorAgent(cfg.llm, plan=plan, pass_threshold=cfg.pass_threshold)

        for round_idx in range(1, cfg.max_rework_rounds + 1):
            eval_batch = [
                (str(o.id), o.question, o.answer) for o in output_items if o.answer
            ]
            eval_results = await evaluator.run_batch(eval_batch, concurrency=cfg.llm.concurrency)

            eval_map: dict[str, EvalResult] = {str(r.item_id): r for r in eval_results}
            for o in output_items:
                if str(o.id) in eval_map:
                    o.evaluation = eval_map[str(o.id)]

            failed = [o for o in output_items if o.evaluation and not o.evaluation.passed]
            passed_count = len(output_items) - len(failed)

            console.print(
                f"  第 {round_idx} 轮评审: "
                f"[green]{passed_count} 通过[/green] / "
                f"[red]{len(failed)} 未通过[/red]"
            )

            if not failed or round_idx == cfg.max_rework_rounds:
                break

            console.print(f"  打回 {len(failed)} 条进行重写...")
            for o in failed:
                feedback = o.evaluation.suggestion if o.evaluation else "请提升回答质量"
                rework_item = CorpusItem(id=o.id, question=o.question, answer=o.answer)
                new_answer = await processor.rework(rework_item, feedback)
                o.answer = new_answer
                o.rework_count += 1

        # ── Stage 4: Aggregator 整合输出 ───────────────────────
        console.print("\n[bold]▶ Stage 4:[/bold] Aggregator 整合输出...")
        aggregator = AggregatorAgent(cfg.llm)
        report = await aggregator.run(output_items, cfg.output_path, cfg.report_path)

        self._display_report(report)
        return report

    # ── 评估模式流水线 ─────────────────────────────────────────
    async def _run_evaluate_pipeline(
        self,
        items: list[CorpusItem],
        plan: PlannerOutput,
    ) -> dict:
        cfg = self.config

        # ── Stage 2: 跳过（评估模式不生成） ────────────────────
        console.print("\n[bold]▶ Stage 2:[/bold] [dim]跳过（评估模式无需生成）[/dim]")

        # ── Stage 3: Evaluator 直接评分 ────────────────────────
        console.print("\n[bold]▶ Stage 3:[/bold] Evaluator 逐条评分...")
        evaluator = EvaluatorAgent(cfg.llm, plan=plan, pass_threshold=cfg.pass_threshold)

        eval_batch = [(str(item.id), item.question, item.answer) for item in items]
        eval_results = await evaluator.run_batch(eval_batch, concurrency=cfg.llm.concurrency)

        eval_map = {str(r.item_id): r for r in eval_results}
        output_items = [
            OutputItem(
                id=item.id,
                question=item.question,
                answer=item.answer,
                evaluation=eval_map.get(str(item.id)),
            )
            for item in items
        ]

        # ── Stage 4: Aggregator 整合报告 ───────────────────────
        console.print("\n[bold]▶ Stage 4:[/bold] Aggregator 整合报告...")
        aggregator = AggregatorAgent(cfg.llm)
        report = await aggregator.run(output_items, cfg.output_path, cfg.report_path)

        self._display_report(report)
        return report

    # ── 辅助方法 ───────────────────────────────────────────────
    async def _load_input(self, user_instruction: str) -> list[CorpusItem]:
        """智能加载输入文件，支持 JSONL/TXT/Word。非结构化文件用 LLM 提取 QA。"""
        result = load_file(self.config.input_path)

        if isinstance(result, list):
            console.print(f"  格式: JSONL（结构化，直接解析）")
            return result

        raw_text = result
        fmt = self.config.input_path.suffix.lower()
        console.print(f"  格式: {fmt}（非结构化，使用 LLM 智能提取 QA 对...）")

        from agents.base import BaseAgent
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
        )

        prompt = build_extraction_prompt(raw_text)
        resp = await client.chat.completions.create(
            model=self.config.llm.model,
            messages=[
                {"role": "system", "content": "你是一个专业的文本解析助手。请严格按要求输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        raw_output = resp.choices[0].message.content or "[]"
        try:
            items = parse_extracted_json(raw_output)
        except Exception as e:
            console.print(f"  [yellow]警告: LLM 提取结果解析失败 ({e})，尝试容错解析...[/yellow]")
            items = self._fallback_parse(raw_text)

        console.print(f"  成功提取 [green]{len(items)}[/green] 个 QA 对")
        return items

    @staticmethod
    def _fallback_parse(text: str) -> list[CorpusItem]:
        """当 LLM 提取失败时的正则兜底解析。"""
        import re
        patterns = [
            r'(?:问题?|Q|Question)\s*[:：]\s*(.+?)(?:\n|$)\s*(?:答案?|A|Answer)\s*[:：]\s*(.+?)(?:\n\n|\n(?=问题?|Q|Question)|$)',
            r'(\d+)\s*[.、)]\s*(.+?)(?:\n|$)\s*(?:答案?|A|Answer)\s*[:：]\s*(.+?)(?:\n\n|\n(?=\d+\s*[.、)])|$)',
        ]

        items: list[CorpusItem] = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                for idx, match in enumerate(matches):
                    if len(match) == 2:
                        items.append(CorpusItem(id=str(idx + 1), question=match[0].strip(), answer=match[1].strip()))
                    elif len(match) == 3:
                        items.append(CorpusItem(id=match[0].strip(), question=match[1].strip(), answer=match[2].strip()))
                break

        return items

    async def _run_with_progress(
        self,
        processor: ProcessorAgent,
        items: list[CorpusItem],
        label: str,
    ) -> dict[str | int, str]:
        """带进度条的并发批量处理。"""
        semaphore = asyncio.Semaphore(self.config.llm.concurrency)
        results: dict[str | int, str] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"  {label}", total=len(items))

            async def _process(item: CorpusItem) -> None:
                async with semaphore:
                    answer = await processor.run(item)
                    results[item.id] = answer
                    progress.advance(task)

            await asyncio.gather(
                *[_process(item) for item in items],
                return_exceptions=True,
            )

        console.print(f"  完成: [green]{len(results)}/{len(items)}[/green]")
        return results

    def _display_plan(self, plan: PlannerOutput) -> None:
        table = Table(title="任务画像", show_header=False, padding=(0, 2))
        table.add_column("属性", style="bold cyan")
        table.add_column("值")
        table.add_row("任务类型", plan.task_type.value)
        table.add_row("核心领域", plan.domain.value)
        table.add_row("约束条件", "\n".join(f"• {c.name}: {c.description}" for c in plan.constraints))
        table.add_row("质量标准", plan.gold_standard[:200] + "..." if len(plan.gold_standard) > 200 else plan.gold_standard)
        console.print(table)

    def _display_report(self, report: dict) -> None:
        console.print()
        table = Table(title="执行报告", show_header=False, padding=(0, 2))
        table.add_column("指标", style="bold cyan")
        table.add_column("值")
        table.add_row("总条目数", str(report.get("total_items", 0)))
        table.add_row("格式异常", str(report.get("items_with_errors", 0)))
        if "avg_score" in report:
            table.add_row("平均评分", f"{report['avg_score']:.2f} / 10")
            table.add_row("最高/最低", f"{report['max_score']:.1f} / {report['min_score']:.1f}")
            table.add_row("通过率", report.get("pass_rate", "N/A"))
            rework = report.get("rework_stats", {})
            table.add_row("重写统计", f"{rework.get('items_reworked', 0)} 条, 共 {rework.get('total_reworks', 0)} 次")
        console.print(table)
        console.print(f"\n[bold green]✓ 输出文件:[/bold green] {self.config.output_path}")
        if self.config.report_path:
            console.print(f"[bold green]✓ 评估报告:[/bold green] {self.config.report_path}")
