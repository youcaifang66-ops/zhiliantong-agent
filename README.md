# 智扫通机器人智能客服

## 项目概述

智扫通机器人智能客服是一个基于大语言模型（LLM）、检索增强生成（RAG）和智能体（Agent）技术的智能客服系统。该系统能够根据用户的问题，结合知识库中的信息，提供准确、专业的回答，特别针对扫地机器人相关的问题。

## 核心功能

- **智能问答**：基于RAG技术，结合知识库内容回答用户问题
- **多工具集成**：集成天气查询、用户位置获取、外部数据获取等多种工具
- **流式响应**：提供实时的流式回答，提升用户体验
- **上下文管理**：维护对话历史，支持多轮对话
- **数据驱动**：可通过扩展知识库内容提升系统回答质量

## 技术栈

- **前端**：Streamlit
- **后端**：Python
- **大语言模型**：通义千问 (Tongyi Qianwen)
- **框架**：LangChain
- **向量数据库**：ChromaDB
- **配置管理**：YAML

## 环境要求

- Python 3.10+
- pip 21.0+

## 安装与配置

### 1. 克隆项目

```bash
git clone <repository-url>
cd AI大模型RAG与智能体开发_Agent项目
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

需要配置通义千问API密钥，可在环境变量中设置：

```bash
# Windows
t set DASHSCOPE_API_KEY=your_api_key

# Linux/macOS
export DASHSCOPE_API_KEY=your_api_key
```

### 4. 配置文件

项目的主要配置文件位于 `config/` 目录：

- `rag.yml`：RAG相关配置
- `agent.yml`：智能体相关配置
- `prompts.yml`：提示词相关配置
- `chroma.yml`：向量数据库相关配置

## 使用指南

### 启动应用

```bash
streamlit run app.py
```

### 基本使用

1. 打开浏览器访问 `http://localhost:8501`
2. 在聊天输入框中输入您的问题，例如：
   - "小户型适合哪些扫地机器人？"
   - "扫地机器人的维护保养方法有哪些？"
   - "如何选择适合自己的扫地机器人？"
3. 系统会实时生成回答，并显示在聊天界面中

### 示例代码

```python
from agent.react_agent import ReactAgent

# 初始化智能体
gent = ReactAgent()

# 执行查询
for chunk in agent.execute_stream("给我生成我的使用报告"):
    print(chunk, end="", flush=True)
```

## 项目结构

```
AI大模型RAG与智能体开发_Agent项目/
├── agent/                 # 智能体相关代码
│   ├── tools/            # 工具函数
│   └── react_agent.py    # 智能体实现
├── config/               # 配置文件
│   ├── agent.yml         # 智能体配置
│   ├── chroma.yml        # 向量数据库配置
│   ├── prompts.yml       # 提示词配置
│   └── rag.yml           # RAG配置
├── data/                 # 数据文件
│   ├── external/         # 外部数据
│   └── 故障排除.txt       # 故障排除文档
├── logs/                 # 日志文件
├── model/                # 模型相关代码
│   └── factory.py        # 模型工厂
├── prompts/              # 提示词文件
├── rag/                  # RAG相关代码
│   ├── chroma_db/        # 向量数据库
│   ├── rag_service.py    # RAG服务
│   └── vector_store.py   # 向量存储
├── utils/                # 工具函数
│   ├── config_handler.py # 配置处理
│   ├── file_handler.py   # 文件处理
│   ├── logger_handler.py # 日志处理
│   ├── path_tool.py      # 路径工具
│   └── prompt_loader.py  # 提示词加载
├── app.py                # 主应用
└── README.md             # 项目说明
```

## 核心模块说明

### 1. 智能体模块 (agent/)

- **ReactAgent**：基于React模式的智能体实现，集成多种工具和中间件
- **工具函数**：包括RAG总结、天气查询、用户位置获取等

### 2. RAG模块 (rag/)

- **RagSummarizeService**：实现检索增强生成功能
- **VectorStoreService**：向量存储服务，用于知识库检索

### 3. 模型模块 (model/)

- **ChatModelFactory**：聊天模型工厂，创建通义千问模型实例
- **EmbeddingsFactory**：嵌入模型工厂，创建文本嵌入模型实例

### 4. 工具模块 (utils/)

- **config_handler**：配置文件处理
- **file_handler**：文件操作处理
- **logger_handler**：日志处理
- **prompt_loader**：提示词加载

## 知识库管理

项目的知识库位于 `data/` 目录，包含以下文件：

- `故障排除.txt`：扫地机器人故障排除指南
- `维护保养.txt`：扫地机器人维护保养方法
- `选购指南.txt`：扫地机器人选购建议

您可以通过添加或修改这些文件来扩展知识库内容。

## 贡献指南

1. Fork本项目
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个Pull Request

## 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。

## 联系方式

- 项目维护者：[您的名字]
- 邮箱：[您的邮箱]
- GitHub：[您的GitHub账号]

## 免责声明

本项目仅用于学习和研究目的，不用于商业用途。使用本项目时，请遵守相关法律法规和服务条款。

---

**更新时间**：2026-04-20