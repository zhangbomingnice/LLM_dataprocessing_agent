# 🧹 Corpus Cleaner — LLM 训练数据清洗 & 标注小助手

> **"你负责训练大模型，我负责帮你洗数据。"**
>
> 一个纯粹为 LLM SFT（监督微调）阶段清洗数据、标注数据而生的小 Agent。
> 不炼丹，不推理，不部署——只搞数据。

---

## 这玩意儿是干嘛的？

你在做大模型微调，面前有一堆质量参差不齐的 QA 数据。有的答案就写了俩字，有的格式乱七八糟，有的甚至答非所问。

你想：
- 🤔 "要是有人帮我把这些破答案全部重写成高质量版本就好了"
- 🤔 "要是有人帮我逐条打个分、标注一下哪些能用哪些得扔就好了"
- 🤔 "要是我扔个 Word 文档进去它就能自动识别里面的 QA 对就好了"
- 🤔 "要是能自动生成带分步推导的 CoT 训练数据就好了"

**恭喜，这就是那个 "有人"。**

Corpus Cleaner 是一个多 Agent 流水线，它接入你选择的大模型 API，按照你指定的标准，自动帮你完成语料的重写、扩充、评估、CoT 标注和打分。生成的不满意？自动打回重写。评分不一致？自动仲裁。数学计算拿不准？SymPy 帮你验算。

## 完整流水线

```
  输入（JSONL / TXT / Word）
         │
         ▼
┌─ 数据预处理 ──────────────────────────────┐
│  --dedup            精确 + 近似去重        │
│  --difficulty       难度分级 (easy/med/hard)│
│  --augment          数据增强（生成变体题）  │
└───────────────────────────────────────────┘
         │
         ▼
┌─ 核心处理（三选一）────────────────────────┐
│  generate    答案重写 / 生成               │
│  evaluate    四维度质量评估                 │
│  cot         CoT 思维链推理标注            │
└───────────────────────────────────────────┘
         │
         ▼
┌─ 质量保障层 ──────────────────────────────┐
│  双重评分验证 + 自动仲裁                   │
│  CoT 逐步骤验证（PRM 思想）               │
│  Self-Consistency 多路采样投票             │
│  SymPy 数学验算引擎                       │
│  不合格 → 自动打回重写                    │
└───────────────────────────────────────────┘
         │
         ▼
  输出（JSONL / TXT / Word）+ 评估报告
```

## 三种核心模式

### 📝 `generate` — 语料生成 / 重写

把你那些简陋的、敷衍的、一句话糊弄人的答案，重写成详细的、分步的、让人看了就想点赞的高质量回答。不合格的自动打回重写，最多循环 N 轮直到满意为止。

### 📊 `evaluate` — 语料质量评估

不改数据，只打分。从「准确性」「完备性」「逻辑严谨性」「格式依从性」四个维度逐条评审，给出分数和改进建议。

### 🧠 `cot` — CoT 思维链标注

**专为训练 LLM 推理能力设计。** 给定数学/物理/逻辑题目，自动生成带步骤标签的 Chain-of-Thought 推理链。每步标注类型（审题/建模/公式推导/代入计算/验证/结论），然后由 CoT-Evaluator **逐步骤验证**——不是整体打分，而是检查每一步的计算、逻辑和公式是否正确。错误步骤精确定位并打回重写。

## 数据预处理工具箱

| 功能 | 开关 | 说明 |
|------|------|------|
| **去重** | `--dedup` | 精确去重 + N-gram Jaccard 近似去重，消灭重复和高度相似的条目 |
| **难度分级** | `--difficulty` | 自动标注 easy / medium / hard，基于关键词、公式密度、题目复杂度 |
| **数据增强** | `--augment` | 用 LLM 从每道题生成 N 个变体（数值替换、条件变换、同类衍生） |

## 质量保障体系

| 防线 | 说明 | 原理 |
|------|------|------|
| **双重评分验证** | 每条答案评两次，分差 > 2 分自动触发第三次仲裁 | 消除 LLM 评分随机性 |
| **CoT 逐步骤验证** | 对推理链的每一步独立验证计算和逻辑 | Process Reward Model 思想（OpenAI） |
| **Self-Consistency** | 同题生成 N 条推理路径，多数投票选答案 | Wang et al., 2022 |
| **SymPy 数学验算** | 把关键数学表达式丢给 SymPy 做符号计算验证 | 形式化验证，100% 确定性 |
| **打回重写循环** | 不合格的条目携带改进建议打回 Processor 重写 | 闭环质控 |

## 快速开始

### 1. 安装

```bash
git clone https://github.com/zhangbomingnice/LLM_dataprocessing_agent.git
cd LLM_dataprocessing_agent
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
```

<details>
<summary>其他模型配置示例</summary>

**DeepSeek（性价比之选，128K 上下文）：**
```env
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
```

**Qwen（通义千问）：**
```env
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-max
```

**本地 Ollama（免费离线）：**
```env
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL_NAME=qwen2.5:72b
OPENAI_API_KEY=ollama
```
</details>

### 3. 运行

```bash
# 重写答案
python main.py generate -i data.jsonl -o output.jsonl "把答案重写为详细长回答"

# 评估质量
python main.py evaluate -i data.jsonl -o scored.jsonl "评估回答质量"

# CoT 思维链标注
python main.py cot -i math.jsonl -o cot_data.jsonl "生成数学分步推导"

# 全套组合：去重 + 难度分级 + CoT + 多路验证
python main.py cot -i math.jsonl -o cot_data.jsonl \
  --dedup --difficulty --self-consistency \
  "生成严格的数学分步推导"

# 数据增强：500 题 → 2000 题
python main.py generate -i data.jsonl -o augmented.jsonl \
  --augment --augment-n 3 \
  "重写为高质量训练数据"

# 从 Word 输入，输出到 Word
python main.py generate -i raw.docx -o polished.docx "重写为学术风格"
```

详细用法见 👉 [**操作手册 MANUAL.md**](./MANUAL.md)

## 输入格式

**JSONL（推荐）：** 每行一个 JSON

```json
{"id": "001", "question": "什么是反向传播？", "answer": "一种训练算法。"}
```

**TXT / Word：** 随便写，只要能看出问题和答案，LLM 自动识别提取。

## 完整参数

```
python main.py {generate,evaluate,cot} "你的需求" [OPTIONS]

核心选项:
  -i, --input           输入文件 (.jsonl/.txt/.docx)
  -o, --output          输出文件 (.jsonl/.txt/.docx)

预处理:
  --dedup               启用去重
  --difficulty          启用难度分级
  --augment             启用数据增强
  --augment-n N         每题变体数 (默认 3)

质量保障:
  --threshold FLOAT     通过阈值 0-10 (默认 7.0)
  --max-rework N        最大重写轮次 (默认 2)
  --self-consistency    启用多路采样验证
  --consistency-n N     采样数量 (默认 5)

模型:
  --model NAME          模型名称
  --base-url URL        API 地址
  --api-key KEY         API Key
  --concurrency N       并发数 (默认 5)
  --max-tokens N        最大 token (默认 4096)
  --temperature FLOAT   生成温度 (默认 0.7)
```

## 项目结构

```
├── main.py                 CLI 入口
├── config.py               配置管理
├── pipeline.py             流水线编排引擎
├── agents/
│   ├── base.py             Agent 基类 + LLM 调用
│   ├── planner.py          规划者：分析需求，动态生成 Prompt
│   ├── processor.py        执行者：并发生成/重写答案
│   ├── evaluator.py        评审员：四维度双重验证评分
│   ├── aggregator.py       整合者：校验 + 多格式输出 + 报告
│   ├── cot_processor.py    CoT 处理器：生成带标签的推理链
│   └── cot_evaluator.py    CoT 评估器：逐步骤验证
├── utils/
│   ├── schema.py           数据模型（Pydantic）
│   ├── io.py               JSONL 读写
│   ├── file_parser.py      多格式输入（JSONL/TXT/Word）
│   ├── file_writer.py      多格式输出（JSONL/TXT/Word）
│   ├── self_consistency.py Self-Consistency 多路采样
│   ├── math_verifier.py    SymPy 数学验算引擎
│   ├── dedup.py            去重（精确 + 近似）
│   ├── difficulty.py       难度分级
│   └── augmentor.py        数据增强
└── examples/               示例数据
```

## 适用场景

- ✅ LLM SFT 训练前的数据清洗
- ✅ 大批量 QA 语料的质量评估与标注
- ✅ 低质量答案的批量重写升级
- ✅ CoT 推理数据标注（数学/物理/逻辑推导）
- ✅ 训练数据去重和质量过滤
- ✅ 题库扩充（数据增强）
- ✅ 数据标注团队的自动化质检
- ❌ 不适合：让它帮你训练模型（它只洗数据，不炼丹）
- ❌ 不适合：当搜索引擎用（它又不是 Google）

---

## 🏆 cn_eval — 中文长回答 SFT 评测系统

> 洗完数据还不够？来，评测一下你训练出来的模型到底行不行。

`cn_eval` 是本项目的第二大模块，专门针对 **中文长回答 SFT 模型** 设计的全自动评测系统。对接 MiniMax API 作为 Judge，支持多种评测模式和深度分析。

### 四种评测模式

| 模式 | 说明 |
|------|------|
| **Pairwise** | 两个模型版本配对对比，Blind A/B 消除位置偏见 |
| **Long-Answer** | 长回答专项：六维度打分 + n-gram 重复率 + 模板化检测 |
| **IF-Eval** | 指令遵循硬验证：正则提取约束 + LLM 辅助判断 |
| **Benchmark** | 基准评测：Exact-Match / F1 / ROUGE-L |

### 核心特性

- **多 Judge 机制**: LLM Judge + Rule Judge + Blind A/B 位置交换 + 多 Judge 聚合
- **稳健统计**: Bootstrap 置信区间、Wilcoxon 符号秩检验、Cohen's d 效应量
- **异常检测**: 长度离群、重复率异常、Judge 不一致、极端分数自动标记
- **长回答深度分析**: 逐句重复模式、前后半段一致性、模板化/助手化诊断
- **版本对比**: 跨版本维度级统计检验 + 效应量分析
- **自动报告**: Markdown 报告 + CSV 数据表 + 雷达图/柱状图/饼图

### 快速开始

```bash
# 使用示例数据评测
python -m cn_eval.cli --config examples/cn_eval/eval_config.yaml

# 使用预设配置
python -m cn_eval.cli --preset pairwise \
  --test-set data/prompts.jsonl \
  --baseline base:data/base.jsonl \
  --candidate v2:data/v2.jsonl

# 仅做长回答评测
python -m cn_eval.cli --preset long_answer \
  --test-set data/prompts.jsonl \
  --baseline base:data/base.jsonl

# 指定模型和并发数
python -m cn_eval.cli --config my_config.yaml \
  --model MiniMax-Text-01 --concurrency 10
```

### 输出结构

```
outputs/
├── eval_results.json      # 完整评测结果
├── report.md              # Markdown 评测报告
├── csv/
│   ├── pairwise.csv       # Pairwise 逐条结果
│   ├── long_answer_*.csv  # 长回答逐条结果
│   ├── anomalies.csv      # 异常样本
│   └── version_compare.csv
└── charts/
    ├── radar.png           # 维度雷达图
    ├── bar_compare.png     # 维度柱状图
    └── winrate_pie.png     # 胜率饼图
```

---

## 技术栈

- **Python 3.10+** + asyncio 异步并发
- **OpenAI SDK** — 兼容 OpenAI / MiniMax / DeepSeek 等 API
- **Pydantic v2** — 数据校验
- **Rich** — 终端进度条、彩色表格
- **SymPy** — 符号数学验算
- **python-docx** — Word 读写
- **PyYAML** — 配置管理
- **matplotlib** — 评测图表
- **jieba** — 中文分词

## License

MIT — 随便用，拿去洗数据 + 评测模型吧。

---

> *"数据质量决定模型上限。别让垃圾数据毁了你的大模型。"*
>
> *—— 某个在凌晨三点手动标注数据后怒写此工具的工程师*
