"""
语料工程多 Agent 流水线 — CLI 入口

用法:
  python main.py generate -i input.jsonl -o output.jsonl "把所有答案重写为分步推导格式"
  python main.py evaluate -i input.jsonl -o scored.jsonl "评估这批数学语料的质量"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from config import PipelineConfig, LLMConfig
from pipeline import Pipeline

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="语料工程多 Agent 流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "mode",
        choices=["generate", "evaluate", "cot"],
        help="运行模式: generate(生成/重写), evaluate(评估打分), cot(思维链标注)",
    )
    parser.add_argument(
        "instruction",
        help="用自然语言描述你的需求，例如'把答案重写为分步推导格式'",
    )
    parser.add_argument("-i", "--input", required=True, help="输入文件路径 (支持 .jsonl / .txt / .docx)")
    parser.add_argument("-o", "--output", default="output.jsonl", help="输出文件路径 (支持 .jsonl / .txt / .docx，按后缀自动选格式)")
    parser.add_argument("-r", "--report", default="report.json", help="评估报告 JSON 路径")
    parser.add_argument("--model", default=None, help="覆盖 LLM 模型名称")
    parser.add_argument("--base-url", default=None, help="覆盖 LLM API base URL")
    parser.add_argument("--api-key", default=None, help="覆盖 API Key")
    parser.add_argument("--concurrency", type=int, default=None, help="并发数 (默认 5)")
    parser.add_argument("--max-tokens", type=int, default=None, help="最大 token 数")
    parser.add_argument("--temperature", type=float, default=None, help="生成温度")
    parser.add_argument("--threshold", type=float, default=7.0, help="通过评分阈值 (默认 7.0)")
    parser.add_argument("--max-rework", type=int, default=2, help="最大重写轮次 (默认 2)")
    parser.add_argument("--dedup", action="store_true", help="启用去重（精确+近似）")
    parser.add_argument("--dedup-threshold", type=float, default=0.85, help="近似去重相似度阈值 (默认 0.85)")
    parser.add_argument("--difficulty", action="store_true", help="启用难度分级标注")
    parser.add_argument("--augment", action="store_true", help="启用数据增强（生成变体题）")
    parser.add_argument("--augment-n", type=int, default=3, help="每题生成变体数 (默认 3)")
    parser.add_argument("--self-consistency", action="store_true", help="启用 Self-Consistency 多路采样验证")
    parser.add_argument("--consistency-n", type=int, default=5, help="多路采样数量 (默认 5)")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")
    return parser


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)

    llm_config = LLMConfig()
    if args.model:
        llm_config.model = args.model
    if args.base_url:
        llm_config.base_url = args.base_url
    if args.api_key:
        llm_config.api_key = args.api_key
    if args.concurrency:
        llm_config.concurrency = args.concurrency
    if args.max_tokens:
        llm_config.max_tokens = args.max_tokens
    if args.temperature is not None:
        llm_config.temperature = args.temperature

    if not llm_config.api_key:
        console.print("[bold red]错误:[/bold red] 未设置 API Key。请配置 .env 文件或使用 --api-key 参数。")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[bold red]错误:[/bold red] 输入文件不存在: {input_path}")
        sys.exit(1)

    config = PipelineConfig(
        llm=llm_config,
        input_path=input_path,
        output_path=Path(args.output),
        report_path=Path(args.report),
        mode=args.mode,
        pass_threshold=args.threshold,
        max_rework_rounds=args.max_rework,
        enable_dedup=args.dedup,
        dedup_threshold=args.dedup_threshold,
        enable_difficulty=args.difficulty,
        enable_augment=args.augment,
        augment_variants=args.augment_n,
        enable_self_consistency=args.self_consistency,
        consistency_samples=args.consistency_n,
    )

    pipeline = Pipeline(config)

    try:
        report = asyncio.run(pipeline.run(args.instruction))
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]流水线执行失败:[/bold red] {e}")
        logging.exception("Pipeline error")
        sys.exit(1)

    console.print("\n[bold green]流水线执行完成！[/bold green]")


if __name__ == "__main__":
    main()
