from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .schema import CorpusItem, OutputItem


def read_jsonl(path: Path) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "id" not in data:
                data["id"] = idx
            items.append(CorpusItem(**data))
    return items


def write_jsonl(items: list[OutputItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json(ensure_ascii=False) + "\n")


def write_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_jsonl(path: Path) -> Iterator[CorpusItem]:
    """内存友好的流式读取。"""
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "id" not in data:
                data["id"] = idx
            yield CorpusItem(**data)
