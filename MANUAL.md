# 📖 操作手册 — Corpus Cleaner 使用指南

## 目录

1. [环境准备](#1-环境准备)
2. [配置说明](#2-配置说明)
3. [输入文件准备](#3-输入文件准备)
4. [运行命令详解](#4-运行命令详解)
5. [两种模式详解](#5-两种模式详解)
6. [评分维度与权重](#6-评分维度与权重)
7. [自定义评价标准](#7-自定义评价标准)
8. [输出文件说明](#8-输出文件说明)
9. [模型选择建议](#9-模型选择建议)
10. [常见问题 FAQ](#10-常见问题-faq)

---

## 1. 环境准备

### 系统要求

- Python 3.10 或更高版本
- 可用的 LLM API（OpenAI / DeepSeek / Qwen / 本地 Ollama 等）

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖清单：
- `openai` — LLM API 调用
- `pydantic` — 数据模型校验
- `python-dotenv` — 环境变量管理
- `rich` — 终端美化输出
- `tenacity` — API 重试机制
- `tiktoken` — Token 计数
- `python-docx` — Word 文件读写

---

## 2. 配置说明

### 创建配置文件

```bash
cp .env.example .env
```

### 配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | 你的 API Key | （必填） |
| `OPENAI_BASE_URL` | API 地址 | `https://api.openai.com/v1` |
| `MODEL_NAME` | 模型名称 | `gpt-4o` |
| `MAX_RETRIES` | API 调用失败重试次数 | `3` |
| `MAX_TOKENS` | 单次生成最大 token 数 | `4096` |
| `TEMPERATURE` | 生成温度（越高越有创意） | `0.7` |
| `EVAL_TEMPERATURE` | 评估温度（越低越稳定） | `0.3` |
| `CONCURRENCY` | 并发请求数 | `5` |

### 不同模型的配置示例

**OpenAI GPT-4o（推荐用于高质量生成）：**

```env
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
```

**DeepSeek（性价比之选，长上下文）：**

```env
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
```

**Qwen（通义千问）：**

```env
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-max
```

**本地 Ollama（免费，离线可用）：**

```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL_NAME=qwen2.5:72b
```

> 💡 **提示：** 你也可以不修改 `.env`，直接在命令行用 `--api-key`、`--base-url`、`--model` 参数覆盖。

---

## 3. 输入文件准备

### 方式 A：JSONL 文件（推荐）

最标准的输入格式。每行一个 JSON 对象，至少包含 `question` 字段：

```json
{"id": "001", "question": "什么是梯度下降？", "answer": "一种优化算法。"}
{"id": "002", "question": "Transformer 的核心机制是什么？", "answer": "注意力机制。"}
{"id": "003", "question": "解释 RLHF 的原理。"}
```

- `id`：可选，不填会自动生成
- `question`：必填
- `answer`：可选。生成模式下如果有原始答案，会在其基础上改进；没有则从零生成

### 方式 B：TXT 纯文本文件

在文本中用自然的格式写出问答对即可：

```
问题：什么是反向传播？
答案：反向传播是一种用于训练神经网络的算法，通过链式法则计算损失函数对每个参数的梯度。

问题：解释 Batch Normalization 的作用。
答案：BN 通过归一化每一层的输入来加速训练。
```

也支持其他常见格式：

```
Q: What is attention mechanism?
A: Attention is a mechanism that allows models to focus on relevant parts of the input.

1. 什么是 Dropout？
答：Dropout 是一种正则化技术。

2. 什么是学习率？
答：控制参数更新步长的超参数。
```

### 方式 C：Word 文档 (.docx)

直接把 Word 文档扔进来。文档中的段落和表格都会被读取，然后由 LLM 自动从中识别和提取 QA 对。

适合场景：你从标注团队或客户那里拿到的原始文档。

---

## 4. 运行命令详解

### 基本格式

```bash
python main.py <模式> -i <输入文件> -o <输出文件> "你的需求描述"
```

### 完整参数表

```
位置参数:
  mode                   运行模式: generate 或 evaluate
  instruction            用自然语言描述你的需求（用引号包裹）

必填选项:
  -i, --input PATH       输入文件路径 (.jsonl / .txt / .docx)

可选选项:
  -o, --output PATH      输出文件路径，默认 output.jsonl
                         （后缀决定格式：.jsonl / .txt / .docx）
  -r, --report PATH      评估报告路径，默认 report.json
  --model NAME           覆盖模型名称
  --base-url URL         覆盖 API 地址
  --api-key KEY          覆盖 API Key
  --concurrency N        并发数，默认 5
  --max-tokens N         最大 token，默认 4096
  --temperature FLOAT    生成温度，默认 0.7
  --threshold FLOAT      通过阈值（0-10），默认 7.0
  --max-rework N         最大重写轮次，默认 2
  -v, --verbose          显示详细日志
```

### 常用命令示例

```bash
# 基本重写
python main.py generate -i data.jsonl -o output.jsonl "重写为详细长回答"

# 数学语料，要求分步推导
python main.py generate -i math_qa.jsonl -o math_output.jsonl "把数学答案重写为严格的分步推导，使用 LaTeX 公式"

# 评估语料质量，输出为 Word
python main.py evaluate -i corpus.jsonl -o quality_report.docx "评估回答质量，重点关注准确性和完备性"

# 从 Word 输入，高并发处理
python main.py generate -i raw.docx -o clean.jsonl --concurrency 10 "整理为高质量 SFT 训练数据"

# 用 DeepSeek 处理，严格阈值
python main.py generate -i data.jsonl -o out.jsonl --model deepseek-chat --base-url https://api.deepseek.com/v1 --threshold 8.0 "重写为学术风格"

# 详细日志模式
python main.py generate -i data.jsonl -o out.jsonl -v "重写答案"
```

---

## 5. 两种模式详解

### 模式 A：generate（生成 / 重写）

**流程：**

1. **Planner** 分析你的需求和数据样本 → 生成专项 Prompt
2. **Processor** 逐条生成新答案（并发处理）
3. **Evaluator** 四维度评分 → 不合格的打回 Processor 重写
4. 重复步骤 2-3，直到全部通过或达到最大重写轮次
5. **Aggregator** 校验格式、整合结果、生成报告

**你的需求描述应该包含：**
- 期望的答案风格（学术、通俗、分步推导等）
- 长度要求（长回答 / 简洁）
- 语言要求（中文 / 英文 / 双语）
- 格式要求（Markdown / LaTeX / 纯文本）

**示例需求描述：**

```
"把所有答案重写为 500 字以上的详细回答，使用 Markdown 格式，包含分步推导，严格中文"
"将英文 QA 翻译并改写为中文 SFT 训练数据，答案要求专业且易懂"
"补全所有缺失的答案，风格参考维基百科，每个答案至少包含 3 个要点"
```

### 模式 B：evaluate（评估 / 打分）

**流程：**

1. **Planner** 根据你的评估要求生成评分标准
2. **Evaluator** 逐条四维度打分 + 双重验证
3. **Aggregator** 整合评分报告（平均分、通过率等）

**你的需求描述应该包含：**
- 你看重哪些维度（准确性 > 格式？逻辑 > 完备性？）
- 领域信息（数学题要看推导过程，代码题要看可运行性）
- 特殊判定标准（比如"答案必须包含公式"）

**示例需求描述：**

```
"评估这批数学语料，重点看解题步骤是否完整、计算是否正确"
"评估代码类 QA 数据，关注代码是否可运行、注释是否清晰"
"评估通用百科数据质量，重点关注事实准确性，格式不重要"
```

---

## 6. 评分维度与权重

Evaluator 默认从四个维度打分，每个维度 0-10 分：

| 维度 | 权重 | 评估内容 |
|------|------|---------|
| **准确性 (Accuracy)** | 35% | 事实是否正确？推导是否无误？ |
| **完备性 (Completeness)** | 25% | 是否完整覆盖问题所有要点？ |
| **逻辑严谨性 (Logic)** | 25% | 论证链条是否连贯？有无逻辑跳跃？ |
| **格式依从性 (Format)** | 15% | 语言、结构、排版是否符合要求？ |

**综合评分** = 准确性 × 0.35 + 完备性 × 0.25 + 逻辑 × 0.25 + 格式 × 0.15

**通过阈值** 默认 7.0（可通过 `--threshold` 调整）。

> 💡 Planner 会根据你的需求自动调整权重侧重。比如你说"重点看格式"，Planner 会在生成的评分 Prompt 中强调格式维度。

---

## 7. 自定义评价标准

**通过自然语言需求来控制。** 你不需要修改代码，只需要在 `instruction` 参数中明确说明你的标准：

```bash
# 重视内容准确性
python main.py evaluate -i data.jsonl -o out.jsonl "严格评估事实准确性，任何事实错误直接判不合格"

# 重视格式规范
python main.py generate -i data.jsonl -o out.jsonl "重写答案，必须使用 Markdown 标题分段，代码用代码块，公式用 LaTeX"

# 重视逻辑深度
python main.py generate -i math.jsonl -o out.jsonl "重写为严格的数学证明格式，每一步推导都要有理由"

# 宽松标准
python main.py evaluate -i data.jsonl -o out.jsonl --threshold 5.0 "基础质检，只要不是明显错误就算通过"

# 严格标准
python main.py evaluate -i data.jsonl -o out.jsonl --threshold 9.0 "最高标准评估，只保留顶级质量的数据"
```

---

## 8. 输出文件说明

### 主输出文件（-o 指定）

根据后缀自动选择格式：

**`.jsonl` — 机器友好**

```json
{"id":"001","question":"什么是梯度下降？","answer":"梯度下降是一种...","evaluation":{"total_score":8.5,"passed":true,"dimensions":[...]}}
```

**`.txt` — 人类可读**

```
============================================================
[001]

【问题】
什么是梯度下降？

【回答】
梯度下降是一种迭代优化算法...

【评分】8.5/10 (通过)
  - 准确性: 9.0 — 事实准确
  - 完备性: 8.0 — 覆盖了主要概念
  ...
```

**`.docx` — Word 文档**

带格式的 Word 文件，包含彩色标题、评分高亮、分页排版，适合发给团队审阅。

### 评估报告（-r 指定）

默认 `report.json`，包含：

```json
{
  "total_items": 100,
  "items_with_errors": 2,
  "scored_items": 100,
  "avg_score": 7.85,
  "max_score": 9.8,
  "min_score": 3.2,
  "pass_count": 82,
  "pass_rate": "82.0%",
  "rework_stats": {
    "total_reworks": 23,
    "items_reworked": 18
  }
}
```

---

## 9. 模型选择建议

| 模型 | 推荐场景 | 优势 | 注意 |
|------|---------|------|------|
| **GPT-4o** | 高质量生成、复杂评估 | 能力最强，格式遵循好 | 贵 |
| **GPT-4o-mini** | 日常使用、大批量处理 | 便宜，速度快 | 复杂推导能力稍弱 |
| **DeepSeek-Chat** | 中文语料、长文本 | 中文能力强，128K 上下文，便宜 | 英文稍弱 |
| **Qwen-Max** | 中文语料 | 中文理解好 | 需要阿里云账号 |
| **Claude 3.5** | 长文本、精细评估 | 200K 上下文，分析能力强 | 需通过兼容 API 接入 |
| **本地模型 (Ollama)** | 隐私敏感、离线场景 | 免费，数据不出本地 | 质量取决于模型大小 |

> 💡 **建议：** 生成用大模型（质量优先），评估可以用小一点的模型（成本优先），因为评分任务相对简单。

---

## 10. 常见问题 FAQ

**Q: 处理速度太慢？**

调高并发数：`--concurrency 10`。注意不要超过 API 的 rate limit。

**Q: 评分不稳定？**

系统已经内置了双重评分验证。如果还不满意，可以降低评估温度（`.env` 中 `EVAL_TEMPERATURE=0.1`）。

**Q: 所有条目都被打回重写？**

可能是阈值设太高了。试试 `--threshold 6.0`。

**Q: Word 文件识别不准？**

Word 中的 QA 对最好有明确的标记（如"问题"/"答案"、"Q"/"A"、编号等），这样 LLM 更容易准确提取。

**Q: API 报错 429 (Rate Limit)？**

降低并发数：`--concurrency 2`。系统内置了自动重试机制（指数退避）。

**Q: 可以用不同模型分别做生成和评估吗？**

当前版本使用统一模型。如果需要分别配置，可以分两次运行：先用大模型 `generate`，再用小模型 `evaluate`。

**Q: 支持增量处理吗？**

目前每次运行是全量处理。如果需要增量，可以把已处理的数据从输入文件中去掉。

---

> 有问题？开 Issue 就好。
