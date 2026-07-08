# MultiAgent DeepResearch

一个基于 **LangGraph + LangChain** 实现的多智能体深度研究系统。项目将复杂研究任务拆解为多个阶段：先生成研究简报和初稿，再由 Supervisor Agent 调度多个 Research Agent 并行检索、压缩研究结果，随后通过 Refinement、Evaluator 和 Red Team 模块对报告进行迭代改写，最后生成结构化研究报告。

> 适合用于学习多智能体协作、LangGraph 状态流转、工具调用、并行 Research Agent、报告评估与自我修正等 Agent 工程流程。

---

## 1. 项目特点

- **多阶段研究流程**：从用户问题到 research brief、draft report、sub-research、refinement、final report。
- **Supervisor 调度机制**：由 Supervisor Agent 决定是否继续检索、是否拆分子任务、是否结束研究。
- **并行 Research Agent**：多个子研究任务可以并行执行，提高复杂问题的信息收集效率。
- **工具调用能力**：支持搜索工具、思考工具、报告精修工具。
- **网页摘要与信息压缩**：对搜索结果和长网页内容进行摘要，降低上下文压力。
- **Evaluator 自评机制**：用评估模型对报告草稿进行综合性、准确性、连贯性打分。
- **Red Team 对抗审查**：专门检查报告中的逻辑漏洞、偏见、不完整之处。
- **角色化模型配置**：不同 Agent 可以使用不同模型，例如 supervisor、writer、researcher、evaluator、red_team 等。

---

## 2. 系统整体流程

```text
用户输入问题
    ↓
write_research_brief
生成研究简报 research_brief
    ↓
write_draft_report
生成初始报告草稿 draft_report
    ↓
supervisor_subgraph
Supervisor Agent 判断需要研究什么
    ↓
ConductResearch 工具调用
并行启动多个 Research Agent
    ↓
Research Agent 搜索、阅读、压缩结果
    ↓
refine_draft_report
根据研究发现修正草稿
    ↓
evaluate_draft_quality
Evaluator 对草稿打分
    ↓
red_team_node
Red Team 找出报告缺陷
    ↓
final_report_generation
生成最终研究报告
```

核心入口在：

```text
deep_research/agent_builder.py
```

其中主图结构为：

```python
deep_researcher_builder.add_edge(START, "write_research_brief")
deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")
deep_researcher_builder.add_edge("write_draft_report", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)
```

---

## 3. 项目目录结构

```text
MultiAgent_DeepResearch/
├── deep_research/
│   ├── agent_builder.py          # 主 LangGraph 构建入口
│   ├── llm.py                    # LLM 客户端初始化与角色模型选择
│   ├── logging.py                # 日志模块
│   ├── utils.py                  # 配置读取、日期等工具函数
│   │
│   ├── agents/
│   │   ├── draft_agent.py        # 研究简报生成与初稿生成
│   │   ├── supervisor.py         # Supervisor Agent，多智能体调度核心
│   │   ├── research_agent.py     # 子研究 Agent，负责搜索与信息压缩
│   │   ├── evaluator_agent.py    # 报告质量评估 Agent
│   │   └── red_team_agent.py     # Red Team 对抗审查 Agent
│   │
│   ├── prompts/
│   │   ├── research_brief.py     # 研究简报提示词
│   │   ├── draft_report.py       # 初稿生成提示词
│   │   ├── research_agent.py     # Research Agent 提示词
│   │   ├── refine_report.py      # 报告修订提示词
│   │   ├── final_report.py       # 最终报告提示词
│   │   ├── red_team.py           # Red Team 提示词
│   │   └── ...
│   │
│   ├── states/
│   │   ├── draft.py              # 主 Agent 状态定义
│   │   ├── supervisor.py         # Supervisor 状态和工具 schema
│   │   ├── research.py           # Research Agent 状态定义
│   │   ├── eval_result.py        # 评估结果结构化输出
│   │   ├── critique.py           # Red Team 批评结构
│   │   └── quality.py            # 质量历史记录结构
│   │
│   ├── tools/
│   │   ├── tool.py               # Tavily search / think / refine 工具
│   │   └── search_factory.py     # 搜索后端工厂
│   │
│   └── providers/
│       └── customsearch.py       # 自定义搜索后端示例
│
├── config.yml                    # 本地运行配置，包含模型角色与搜索后端配置
├── requirements.txt              # Python 依赖
├── test.py                       # 最小运行示例
├── run.ipynb                     # Notebook 调试入口，可选
└── README.md
```

---

## 4. 核心模块说明

### 4.1 `agent_builder.py`

主 Agent 图的入口，负责编排完整研究流程：

1. `write_research_brief`：根据用户输入生成研究简报。
2. `write_draft_report`：基于研究简报生成初始草稿。
3. `supervisor_subgraph`：进入 Supervisor 子图，进行研究调度。
4. `final_report_generation`：基于草稿和研究 notes 生成最终报告。

---

### 4.2 `draft_agent.py`

负责研究流程的前置准备：

- 将用户原始问题改写为更适合研究的 `research_brief`。
- 根据 `research_brief` 生成初始 `draft_report`。
- 将草稿和研究简报传递给 Supervisor。

---

### 4.3 `supervisor.py`

项目最核心的多智能体调度模块。

Supervisor Agent 可以调用以下工具：

- `ConductResearch`：把某个子主题委派给 Research Agent。
- `ResearchComplete`：表示研究完成。
- `think_tool`：记录阶段性思考和决策依据。
- `refine_draft_report`：根据研究结果修正草稿。

Supervisor 会根据当前状态决定：

```text
继续搜索 / 并行启动子 Agent / 修正报告 / 结束研究
```

其中并行研究通过 `asyncio.gather()` 实现。

---

### 4.4 `research_agent.py`

Research Agent 是具体执行检索和信息压缩的子 Agent。

核心循环为：

```text
llm_call
    ↓
是否有 tool_calls？
    ├── 有：tool_node 执行搜索或思考工具，再回到 llm_call
    └── 无：compress_research 压缩研究结果
```

Research Agent 最终输出：

- `compressed_research`：压缩后的高价值研究结论。
- `raw_notes`：原始搜索结果和中间信息。

---

### 4.5 `tools/tool.py`

主要工具包括：

| 工具 | 作用 |
|---|---|
| `tavily_search` | 调用 Tavily 搜索并返回格式化结果 |
| `think_tool` | 让 Agent 显式记录思考、差距评估、下一步计划 |
| `refine_draft_report` | 根据 research brief、findings 和 draft report 修订报告 |

---

### 4.6 `evaluator_agent.py`

用于对报告草稿进行结构化评分，输出：

- `comprehensiveness_score`：综合性得分。
- `accuracy_score`：事实准确性得分。
- `coherence_score`：逻辑与表达连贯性得分。
- `reason`：评分原因和修改建议。

---

### 4.7 `red_team_agent.py`

Red Team Agent 用于发现报告中的：

- 逻辑漏洞
- 偏见
- 证据不足
- 没有充分回答用户问题的部分

如果 Red Team 发现问题，会把批评意见注入 Supervisor 的上下文，让 Supervisor 在后续迭代中修复。

---

## 5. 环境安装

建议使用 Python 3.11 或 3.12。

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 6. 配置方式

项目通过 `config.yml` 配置模型、搜索后端和不同 Agent 的角色模型。

### 6.1 API Key 配置

请不要把真实 API Key 提交到 GitHub。

推荐本地使用 `.env` 或只在本机 `config.yml` 中填写 key，公开仓库中只保留 `config.example.yml`。

`.env.example` 示例：

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=MultiAgent_DeepResearch
```

### 6.2 `config.yml` 示例

```yaml
stages:
  prod:
    cognition:
      openai:
        base_url: https://api.openai.com/v1
        api_key: your_openai_api_key_here
        default_model: gpt-4o-mini
        temperature: 0
        timeout_seconds: 600
        models:
          gpt-4o-mini:
            max_tokens: 128000

    search:
      backend: tavily
      tavily:
        api_key: your_tavily_api_key_here
        base_url: https://api.tavily.com
        max_results: 3
        topic: general
        include_raw_content: true
        timeout_seconds: 300

    roles:
      supervisor:
        backend: openai
        handle: gpt-4o-mini
        timeout_seconds: 600
      researcher_main:
        backend: openai
        handle: gpt-4o-mini
        timeout_seconds: 600
      researcher_summarizer:
        backend: openai
        handle: gpt-4o-mini
        timeout_seconds: 600
      researcher_compressor:
        backend: openai
        handle: gpt-4o-mini
        timeout_seconds: 600
      draft:
        backend: openai
        handle: gpt-4o-mini
        timeout_seconds: 600
      writer:
        backend: openai
        handle: gpt-4o-mini
        timeout_seconds: 600
      evaluator:
        backend: openai
        handle: gpt-4o-mini
        timeout_seconds: 600
      red_team:
        backend: openai
        handle: gpt-4o-mini
        timeout_seconds: 600
```

> 注意：当前代码中的 `load_config()` 直接读取 `config.yml`。如果希望 `config.yml` 自动解析 `${OPENAI_API_KEY}` 这类环境变量，需要额外实现环境变量替换逻辑，或者直接在本地私有 `config.yml` 中填写真实 key。

---

## 7. 运行示例

项目提供了 `test.py` 作为最小运行示例：

```bash
python test.py
```

`test.py` 会调用主图：

```python
from deep_research.agent_builder import deep_researcher_builder
```

示例输入：

```python
HumanMessage(content="简单介绍一下 AI Agent Memory。")
```

运行结束后会输出：

```text
final_report
```

也可以在自己的脚本中这样调用：

```python
import asyncio
from langchain_core.messages import HumanMessage
from deep_research.agent_builder import deep_researcher_builder

async def main():
    agent = deep_researcher_builder.compile()

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="请系统调研 Agent Memory 模块的发展方向。")
            ]
        },
        config={
            "recursion_limit": 50
        }
    )

    print(result["final_report"])

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 8. LangSmith 观测

项目支持通过 LangSmith 观察 Agent 图执行过程、模型调用、工具调用和中间状态。

在 `.env` 中配置：

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=MultiAgent_DeepResearch
```

运行后，可以在 LangSmith 项目中查看：

- 主图节点执行顺序
- Supervisor 工具调用
- Research Agent 搜索过程
- LLM 输入输出
- 报告修订过程

---

## 9. 适合学习的重点

这个项目适合重点学习以下 Agent 工程能力：

1. **LangGraph 状态图设计**
   - `StateGraph`
   - `START` / `END`
   - `add_node`
   - `add_edge`
   - `add_conditional_edges`
   - 子图嵌套

2. **多 Agent 协作**
   - Supervisor 负责规划和分派任务
   - Research Agent 负责执行检索
   - Evaluator 负责质量评分
   - Red Team 负责对抗式审查
   - Writer 负责最终报告生成

3. **工具调用机制**
   - LangChain `@tool`
   - `bind_tools`
   - `tool_calls`
   - `ToolMessage`

4. **状态传递机制**
   - `TypedDict`
   - `MessagesState`
   - `Annotated[..., add_messages]`
   - `operator.add`

5. **工程化配置**
   - 按 role 选择模型
   - 支持搜索后端配置
   - 日志与运行观测
   - 异步并行执行

---

## 10. 隐私与上传 GitHub 注意事项

上传 GitHub 前必须删除或忽略以下内容：

```text
.env
config.yml 中的真实 API Key
__pycache__/
.ipynb_checkpoints/
.vscode/
*.pyc
run.ipynb 的运行输出
```

推荐 `.gitignore`：

```gitignore
# secrets
.env
.env.*
*.key
*.pem
*.token

# Python cache
__pycache__/
*.pyc
.ipynb_checkpoints/

# local editor / system files
.vscode/
.DS_Store
Thumbs.db

# logs
*.log

# model / large files
*.pt
*.pth
*.bin
*.safetensors
*.ckpt
models/
checkpoints/

# archives
*.zip
*.tar
*.tar.gz
*.rar
*.7z
```

建议将真实配置文件改为本地私有文件，并提供公开模板：

```text
config.example.yml
.env.example
```

---

## 11. 已知问题与可改进方向

- 当前 `config.yml` 读取逻辑没有自动展开环境变量，需要手动填写 key 或补充环境变量解析函数。
- `run.ipynb` 适合调试，但公开仓库中建议清除输出或不上传。
- `t.py` 属于临时测试脚本，如包含 key 打印逻辑，应删除。
- Supervisor 的最大迭代次数和最大并行子 Agent 数目前写在代码常量中，可迁移到配置文件。
- 搜索后端目前主要面向 Tavily，可以继续扩展 Google Search、Bing Search、本地知识库或向量数据库。
- Red Team 的反馈目前主要通过上下文注入处理，后续可以增加显式 critique resolved 状态更新。
- 可以增加 CLI 参数，例如：

```bash
python run.py --query "你的研究问题" --output report.md
```

---

## 12. 示例研究主题

可以尝试以下问题：

```text
系统调研 Agent Memory 模块的发展方向，包括短期记忆、长期记忆、检索式记忆和个性化记忆。
```

```text
请研究多智能体系统中的 Supervisor 架构，包括任务拆解、工具调用、失败恢复和评估机制。
```

```text
请从工程落地角度调研 Deep Research Agent 的典型架构、关键模块和未来演进方向。
```

---

## 13. License

本项目主要用于个人学习、课程项目和 Agent 工程实验。公开发布前请确认依赖库 license、API Key 安全性以及是否引用了外部开源项目。
