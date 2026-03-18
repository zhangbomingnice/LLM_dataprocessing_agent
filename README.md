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

**恭喜，这就是那个 "有人"。**

Corpus Cleaner 是一个四阶段 Agent 流水线，它接入你选择的大模型 API，按照你指定的标准，自动帮你完成语料的重写、扩充、评估和打分。生成的不满意？自动打回重写。评分不一致？自动仲裁。你只需要准备好数据，告诉它你想要什么，然后去泡杯咖啡。

## 架构：四个打工 Agent

```
              你的需求 + 数据
                    │
                    ▼
          ┌──────────────────┐
          │   🧠 Planner     │ ← 分析你的需求，给其他三个发活
          │   (规划者)        │
          └────────┬─────────┘
                   │ 动态 Prompt
                   ▼
          ┌──────────────────┐
          │   ✍️ Processor    │ ← 逐条生成/重写答案，并发干活
          │   (执行者)        │
          └────────┬─────────┘
                   │ 生成结果           ╭─ 不合格？打回重写！
                   ▼                    │
          ┌──────────────────┐          │
          │   🔍 Evaluator   │ ← 四维度打分 + 双重验证
          │   (评审员)        │──────────╯
          └────────┬─────────┘
                   │ 评审报告
                   ▼
          ┌──────────────────┐
          │   📦 Aggregator  │ ← 校验格式、整合输出、生成报告
          │   (整合者)        │
          └──────────────────┘
                   │
                   ▼
        output.jsonl / .docx / .txt
```

## 两种工作模式

### 📝 模式 A：语料生成 / 重写

把你那些简陋的、敷衍的、一句话糊弄人的答案，重写成详细的、分步的、让人看了就想点赞的高质量回答。不合格的自动打回重写，最多循环 N 轮直到满意为止。

### 📊 模式 B：语料质量评估

不改数据，只打分。从「准确性」「完备性」「逻辑严谨性」「格式依从性」四个维度逐条评审，给出分数和改进建议。适合数据标注和质检场景。

## 亮点功能

| 功能 | 说明 |
|------|------|
| **多格式输入** | 扔 `.jsonl`、`.txt`、`.docx` 进来都行，非结构化文件自动用 LLM 提取 QA 对 |
| **多格式输出** | 输出后缀写 `.jsonl` / `.txt` / `.docx`，自动切换格式 |
| **模型自选** | 接入任何 OpenAI 兼容 API — GPT-4o、DeepSeek、Qwen、Claude、Ollama 本地模型，你说了算 |
| **动态 Prompt** | Planner 根据你的数据和需求，自动为下游 Agent 生成最优 System Prompt |
| **双重评分验证** | 每条答案评两次，分差过大自动启动第三次仲裁，杜绝随机评分 |
| **打回重写机制** | 评分不达标自动打回 Processor 重写，携带具体改进建议 |
| **异步并发** | 多条数据并行处理，大批量也扛得住 |

## 快速开始

### 1. 安装

```bash
git clone git@github.com:zhangbomingnice/LLM_dataprocessing_agent.git
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

**用 DeepSeek？**

```env
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
```

**用本地 Ollama？**

```env
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL_NAME=qwen2.5:72b
OPENAI_API_KEY=ollama
```

### 3. 运行

```bash
# 重写 — 把答案升级为长回答
python main.py generate -i my_data.jsonl -o output.jsonl "把所有答案重写为详细的分步推导格式"

# 评估 — 给现有 QA 打分
python main.py evaluate -i my_data.docx -o report.txt "评估回答质量，重点看逻辑严谨性"

# Word 进，Word 出
python main.py generate -i raw_qa.docx -o polished.docx "重写为学术风格长回答"
```

详细用法见 👉 [**操作手册 MANUAL.md**](./MANUAL.md)

## 输入格式

**JSONL（推荐）：**

```json
{"id": "001", "question": "什么是反向传播？", "answer": "一种训练算法。"}
```

**TXT / Word：** 随便写，只要能看出问题和答案，LLM 自动识别提取。

## 输出示例

**评估报告 (report.json)：**

```json
{
  "total_items": 100,
  "avg_score": 7.85,
  "pass_rate": "82.0%",
  "rework_stats": { "items_reworked": 18, "total_reworks": 23 }
}
```

## 适用场景

- ✅ LLM SFT 训练前的数据清洗
- ✅ 大批量 QA 语料的质量评估与标注
- ✅ 低质量答案的批量重写升级
- ✅ 数据标注团队的自动化质检
- ❌ 不适合：让它帮你训练模型（它只洗数据，不炼丹）
- ❌ 不适合：当搜索引擎用（它又不是 Google）

## 技术栈

- **Python 3.10+** + asyncio 异步并发
- **OpenAI SDK** — 兼容一切 OpenAI 格式 API
- **Pydantic v2** — 数据校验
- **Rich** — 终端进度条、彩色表格
- **python-docx** — Word 读写

## License

MIT — 随便用，拿去洗数据吧。

---

> *"数据质量决定模型上限。别让垃圾数据毁了你的大模型。"*
>
> *—— 某个在凌晨三点手动标注数据后怒写此工具的工程师*
