"""
统一数据加载器 — 支持 JSONL / CSV / XLSX。

根据文件后缀自动选择解析方式，将原始数据映射为统一的 schema 模型。
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from .schema import Prompt, ModelOutput, ABMapping

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class UnifiedLoader:
    """统一文件加载器。"""

    @staticmethod
    def load_raw(path: Path) -> list[dict[str, Any]]:
        """加载原始文件为 dict 列表。"""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in (".jsonl", ".json"):
            return UnifiedLoader._load_jsonl(path)
        elif suffix == ".csv":
            return UnifiedLoader._load_csv(path)
        elif suffix in (".xlsx", ".xls"):
            return UnifiedLoader._load_xlsx(path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}（支持 .jsonl/.csv/.xlsx）")

    @staticmethod
    def load_as(path: Path, model_class: Type[T], **field_mapping: str) -> list[T]:
        """
        加载文件并映射为指定的 Pydantic 模型。

        field_mapping: 原始字段名 → 模型字段名的映射。
        例如 load_as(path, Prompt, id="prompt_id", question="text")
        """
        raw_items = UnifiedLoader.load_raw(path)
        results: list[T] = []

        for idx, item in enumerate(raw_items):
            mapped = {}
            for raw_key, model_key in field_mapping.items():
                if raw_key in item:
                    mapped[model_key] = item[raw_key]

            for key, val in item.items():
                if key not in field_mapping:
                    mapped.setdefault(key, val)

            try:
                results.append(model_class(**mapped))
            except Exception as e:
                logger.warning("第 %d 行数据解析失败: %s (数据: %s)", idx, e, item)

        logger.info("从 %s 加载 %d 条 %s", path, len(results), model_class.__name__)
        return results

    @staticmethod
    def load_prompts(path: Path, id_field: str = "prompt_id", text_field: str = "text",
                     category_field: str = "category") -> list[Prompt]:
        """加载测试题集。"""
        raw = UnifiedLoader.load_raw(path)
        results = []
        for item in raw:
            results.append(Prompt(
                prompt_id=str(item.get(id_field, "")),
                text=item.get(text_field, ""),
                category=item.get(category_field, ""),
                subset=item.get("subset", ""),
                metadata={k: v for k, v in item.items()
                          if k not in (id_field, text_field, category_field, "subset")},
            ))
        logger.info("加载 %d 条 Prompt ← %s", len(results), path)
        return results

    @staticmethod
    def load_model_outputs(path: Path, model_version: str,
                           id_field: str = "prompt_id",
                           response_field: str = "response") -> list[ModelOutput]:
        """加载单个模型版本的输出。"""
        raw = UnifiedLoader.load_raw(path)
        results = []
        for item in raw:
            results.append(ModelOutput(
                prompt_id=str(item.get(id_field, "")),
                model_version=model_version,
                response=item.get(response_field, ""),
                metadata={k: v for k, v in item.items()
                          if k not in (id_field, response_field)},
            ))
        logger.info("加载 %d 条模型输出 [%s] ← %s", len(results), model_version, path)
        return results

    @staticmethod
    def load_answer_key(path: Path) -> list[ABMapping]:
        """
        加载 A/B 映射文件。

        支持格式:
          {"prompt_001": {"A": "v2.7", "B": "base"}, ...}
        或 JSONL:
          {"prompt_id": "prompt_001", "A": "v2.7", "B": "base"}
        """
        path = Path(path)
        raw = UnifiedLoader.load_raw(path)

        # 如果是单个 JSON dict（非 JSONL），特殊处理
        if len(raw) == 1 and not any("prompt_id" in r for r in raw):
            mapping_dict = raw[0]
            results = []
            for pid, mapping in mapping_dict.items():
                results.append(ABMapping(
                    prompt_id=str(pid),
                    a_model=mapping.get("A", ""),
                    b_model=mapping.get("B", ""),
                ))
            return results

        results = []
        for item in raw:
            results.append(ABMapping(
                prompt_id=str(item.get("prompt_id", "")),
                a_model=item.get("A", ""),
                b_model=item.get("B", ""),
            ))
        logger.info("加载 %d 条 A/B 映射 ← %s", len(results), path)
        return results

    # ── 底层解析 ───────────────────────────────────────────────

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

            # 尝试整体 JSON 解析（单文件 JSON dict/array）
            if content.startswith("{") and not content.startswith('{"'):
                try:
                    return [json.loads(content)]
                except json.JSONDecodeError:
                    pass
            if content.startswith("["):
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass

            # 逐行 JSONL
            for line in content.splitlines():
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    @staticmethod
    def _load_csv(path: Path) -> list[dict]:
        items = []
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(dict(row))
        return items

    @staticmethod
    def _load_xlsx(path: Path) -> list[dict]:
        try:
            import openpyxl
        except ImportError:
            raise ImportError("读取 .xlsx 需要 openpyxl: pip install openpyxl")

        wb = openpyxl.load_workbook(str(path), read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []

        headers = [str(h) if h else f"col_{i}" for i, h in enumerate(rows[0])]
        items = []
        for row in rows[1:]:
            item = {h: (v if v is not None else "") for h, v in zip(headers, row)}
            items.append(item)

        wb.close()
        return items
