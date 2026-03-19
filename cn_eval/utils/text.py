"""
中文文本处理工具 — 分句、分段、分词、n-gram。
"""

from __future__ import annotations

import re
from typing import Iterator

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


def split_sentences(text: str) -> list[str]:
    """中文分句。"""
    pattern = r'([。！？；…\!\?\;])'
    parts = re.split(pattern, text)

    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sent = parts[i] + parts[i + 1]
        sent = sent.strip()
        if sent:
            sentences.append(sent)

    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())

    return sentences


def split_paragraphs(text: str) -> list[str]:
    """按换行分段，过滤空段。"""
    paras = re.split(r'\n{1,}', text)
    return [p.strip() for p in paras if p.strip()]


def tokenize(text: str) -> list[str]:
    """中文分词（jieba）。"""
    if JIEBA_AVAILABLE:
        return list(jieba.cut(text))
    return list(text)


def char_ngrams(text: str, n: int) -> list[str]:
    """字符级 n-gram。"""
    text = re.sub(r'\s+', '', text)
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def word_ngrams(text: str, n: int) -> list[str]:
    """词级 n-gram（需 jieba）。"""
    words = tokenize(text)
    words = [w for w in words if w.strip()]
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]


def ngram_repetition_rate(text: str, n: int = 4, level: str = "word") -> float:
    """
    计算 n-gram 重复率。

    level: "char" 字级 / "word" 词级
    返回: 重复 n-gram 占总 n-gram 的比例 (0~1)
    """
    if level == "char":
        grams = char_ngrams(text, n)
    else:
        grams = word_ngrams(text, n)

    if not grams:
        return 0.0

    unique = set(grams)
    return 1.0 - len(unique) / len(grams)


def count_chars(text: str, exclude_whitespace: bool = True) -> int:
    """统计字符数。"""
    if exclude_whitespace:
        text = re.sub(r'\s', '', text)
    return len(text)


def detect_template_endings(text: str, patterns: list[str] | None = None) -> list[str]:
    """检测模板化结尾。"""
    if patterns is None:
        patterns = [
            "总之", "综上所述", "希望以上内容对你有所帮助",
            "以上就是", "希望对您有帮助", "如果你有其他问题",
            "以上是我的回答", "希望能帮到你",
        ]

    found = []
    last_200 = text[-200:] if len(text) > 200 else text
    for p in patterns:
        if p in last_200:
            found.append(p)
    return found


def detect_assistant_phrases(text: str) -> list[str]:
    """检测助手化用语。"""
    phrases = [
        "您好", "你好", "以下是我的回答", "希望对您有帮助",
        "希望以上内容", "如果您还有", "如果你还有",
        "作为一个AI", "作为AI助手", "很高兴为您",
        "请注意", "需要注意的是", "值得一提的是",
    ]
    found = []
    for p in phrases:
        if p in text:
            found.append(p)
    return found
