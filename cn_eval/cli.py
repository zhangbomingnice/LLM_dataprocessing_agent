"""
cn_eval CLI — 中文长回答 SFT 评测系统命令行入口。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

from cn_eval.engine import EvalEngine
from cn_eval.utils.config import load_config

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cn-eval",
        description="中文长回答 SFT 评测系统（统一 LLM Judge）",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径 (YAML)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["pairwise", "single", "full"],
        help="使用预设配置",
    )
    parser.add_argument("--test-set", type=str, help="测试题集路径")
    parser.add_argument(
        "--baseline", type=str,
        help="baseline 模型输出路径 (格式: version:path)",
    )
    parser.add_argument(
        "--candidate", type=str, action="append",
        help="candidate 模型输出路径 (格式: version:path，可多次)",
    )
    parser.add_argument(
        "--modes", type=str, nargs="+",
        choices=["pairwise", "single"],
        help="评测模式",
    )
    parser.add_argument("--output-dir", "-o", type=str, help="输出目录")
    parser.add_argument("--model", type=str, help="LLM 模型名称")
    parser.add_argument("--api-key", type=str, help="API Key")
    parser.add_argument("--api-base", type=str, help="API Base URL")
    parser.add_argument("--concurrency", type=int, help="并发请求数")
    parser.add_argument("--rounds", type=int, help="一致性评审轮数 (默认 3)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    return parser


def parse_model_path(spec: str) -> tuple[str, str]:
    """解析 version:path 格式，兼容 Windows 盘符。"""
    if ":" in spec and not spec[1] == ":":
        parts = spec.split(":", 1)
        return parts[0], parts[1]
    return "model", spec


def main():
    parser = build_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )

    if args.preset:
        preset_map = {
            "pairwise": "configs/eval_presets/pairwise.yaml",
            "single": "configs/eval_presets/single.yaml",
            "full": "configs/eval_presets/full.yaml",
        }
        config_path = preset_map[args.preset]
    else:
        config_path = args.config

    overrides: dict = {}
    if args.test_set:
        overrides["test_set_path"] = args.test_set
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.modes:
        overrides["eval_modes"] = args.modes

    config = load_config(config_path, overrides)

    if args.model:
        config.judge.primary_model = args.model
    if args.api_key:
        config.judge.primary_api_key = args.api_key
    if args.api_base:
        config.judge.primary_api_base = args.api_base
    if args.concurrency:
        config.judge.concurrency = args.concurrency
    if args.rounds:
        config.consistency.num_rounds = args.rounds

    if args.baseline:
        ver, path = parse_model_path(args.baseline)
        config.baseline = ver
        config.model_outputs[ver] = path

    if args.candidate:
        for spec in args.candidate:
            ver, path = parse_model_path(spec)
            config.model_outputs[ver] = path
            if ver not in config.candidates:
                config.candidates.append(ver)

    engine = EvalEngine(config)

    try:
        results = asyncio.run(engine.run())
        console.print(f"\n[bold green]评测完成，共 {len(results)} 个模式[/bold green]")
    except KeyboardInterrupt:
        console.print("\n[bold red]已中断[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]错误: {e}[/bold red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
