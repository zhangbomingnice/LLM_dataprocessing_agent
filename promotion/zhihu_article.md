# 做 LLM 微调时，我受够了手动清洗数据，所以写了个 Agent 帮我干活

## 前言：数据清洗，大模型微调最没技术含量但最要命的环节

最近在做大模型 SFT 微调，模型选好了，框架搭好了，超参也调了，结果效果就是不行。

排查了一圈发现——**数据有问题。**

几万条 QA 训练数据里面：
- 有的答案就一句话："是的。"（你倒是展开说说啊）
- 有的答案格式乱成一锅粥，Markdown 和纯文本混着来
- 有的答案干脆是错的，张冠李戴
- 还有一些甲方/标注团队交过来的 Word 文档，问答对混在段落里，手动拆分到怀疑人生

于是就进入了经典流程：人工一条条看，一条条改，改到凌晨三点，脖子僵了，眼睛花了，数据还没清完一半。

我就想：**这活儿能不能让 LLM 自己干？**

用 GPT-4o 重写答案，用 DeepSeek 来评分，不合格的自动打回重写，循环往复直到达标——这不就是个流水线吗？

于是我花了点时间写了这么个小工具。

---

## Corpus Cleaner：四个 Agent 组成的数据清洗流水线

核心思路很简单：**用大模型来清洗大模型的训练数据。**

整个系统是四个 Agent 串联的流水线：

```
你的需求 + 数据
      │
      ▼
  🧠 Planner（规划者）
      │  分析你的需求和数据样本
      │  自动生成最优的 System Prompt
      ▼
  ✍️ Processor（执行者）
      │  逐条生成/重写答案
      │  支持并发批量处理
      ▼
  🔍 Evaluator（评审员）
      │  四个维度打分：准确性/完备性/逻辑/格式
      │  双重评分验证，分差大自动仲裁
      │  不合格 → 打回 Processor 重写
      ▼
  📦 Aggregator（整合者）
      │  校验格式、整合结果、输出报告
      ▼
  output.jsonl / .txt / .docx
```

### 两种工作模式

**模式 A：生成/重写**

你有一批答案质量差的 QA 数据，让 Agent 帮你全部重写成高质量版本。评分不过关的自动打回重写，最多循环 N 轮。

```bash
python main.py generate -i data.jsonl -o output.jsonl "把所有答案重写为500字以上的详细分步推导"
```

**模式 B：质量评估**

你不想改数据，只想知道哪些数据能用、哪些得扔。Agent 逐条打分，给出四维度评分和改进建议。

```bash
python main.py evaluate -i data.jsonl -o report.txt "评估回答质量，重点看逻辑严谨性"
```

---

## 几个我觉得比较实用的设计

### 1. Planner 动态生成 Prompt

不是写死一套 Prompt 硬套所有数据。Planner 会先看你的数据样本和需求，自动判断领域（数学/代码/百科/文学），然后为 Processor 和 Evaluator 生成量身定制的 System Prompt。

比如你说"重写数学题答案"，Planner 会自动在 Prompt 里加上"使用 LaTeX 公式"、"分步推导"这类约束。你说"评估代码类 QA"，它会自动侧重"代码可运行性"维度。

### 2. 双重评分验证

评分这件事，LLM 其实不太稳定——同样的 QA 对，评两次可能差个一两分。

所以我做了个双重验证：每条数据用不同温度评两次，分差在 2 分以内取平均，超过 2 分就自动触发第三次"仲裁评分"。实测下来评分一致性提升了不少。

### 3. 多格式输入输出

这个是被逼出来的。标注团队交过来的数据什么格式都有——有 JSONL 的，有 TXT 的，还有 Word 的。

所以系统支持直接扔 `.jsonl`、`.txt`、`.docx` 进来。非结构化文件会用 LLM 自动识别提取 QA 对。输出也一样，后缀写 `.docx` 就出 Word，写 `.txt` 就出纯文本。

### 4. 模型随便换

底层用的 OpenAI 兼容 API，所以 GPT-4o、DeepSeek、Qwen、Claude、本地 Ollama 都能接。

我自己的用法是：**生成用 DeepSeek（便宜，128K 上下文，中文好），评估用 GPT-4o-mini（评分任务简单，用小模型就够）。**

---

## 实际效果

拿了一批 500 条的数学 QA 数据测试：

- 原始数据人工评估：约 40% 的答案质量不合格（过短、不完整、有错误）
- 跑完流水线后：平均评分从 5.2 提升到 8.1，通过率 89%
- 被打回重写的 87 条，其中 71 条在第二轮通过

当然，最终质量还是取决于你用的模型。模型越强，天花板越高。

---

## 如何使用

```bash
# 克隆
git clone https://github.com/zhangbomingnice/LLM_dataprocessing_agent.git
cd LLM_dataprocessing_agent

# 安装
pip install -r requirements.txt

# 配置（填入你的 API Key）
cp .env.example .env

# 运行
python main.py generate -i your_data.jsonl -o output.jsonl "你的需求描述"
```

项目里有详细的操作手册（MANUAL.md），包括各种模型的配置方法、命令参数、评分维度说明等，照着走就行。

**GitHub 地址：** https://github.com/zhangbomingnice/LLM_dataprocessing_agent

---

## 写在最后

这个工具的定位很明确：**就是给做 LLM 微调的人清洗数据用的。** 不炼丹，不推理，不部署，只负责把数据洗干净。

如果你也在被 SFT 数据质量折磨，可以试试。有问题欢迎在 GitHub 提 Issue，或者评论区聊。

觉得有用的话，给个 Star 就是最好的支持了。

---

*（标签建议：大模型、LLM、SFT、微调、数据清洗、数据标注、Agent、开源工具）*
