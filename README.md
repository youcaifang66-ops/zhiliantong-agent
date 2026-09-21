# 智练通（FitFlow Agent）

面向个人训练计划、训练记录与周期复盘的健身 Agent。项目以可解释意图路由、混合检索、分层记忆和工具调用为核心，逐步建设一套可评测、可恢复的 Agent 工程系统。

> 当前版本：`v0.1.0`，完成健身领域 MVP。训练记录持久化、LangGraph 状态机、引用溯源和完整评测将在后续里程碑实现，详见仓库路线图。

## MVP 能力

- 五类意图：知识问答、计划制定、训练记录、训练调整、训练报告。
- BM25 + Embedding + RRF 混合检索，知识库覆盖动作、计划原则和训练安全。
- 独立场景提示词与动态中间件路由。
- 训练历史查询、天气、位置、用户与月份等工具调用。
- 短期窗口、向量记忆和摘要记忆的原始基础实现。
- FastAPI 流式与同步接口，以及最小 Web 对话界面。

## 快速开始

1. 安装 Python 3.11 或 3.12，创建虚拟环境并安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. 复制环境变量示例并配置模型服务：

   ```bash
   cp .env.example .env
   ```

3. 启动服务：

   ```bash
   python fastapi_main.py
   ```

4. 打开 `http://localhost:8001`，接口文档位于 `/docs`。

## API 示例

```bash
curl -X POST http://localhost:8001/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"user_id":"1001","query":"帮我制定一个每周三练的增肌计划"}'
```

健康检查：`GET /health`。流式聊天：`POST /chat`，响应格式为 SSE。

## 版本路线

完整的十次迭代目标见 [docs/ROADMAP.md](docs/ROADMAP.md)。每一阶段只声明已经实现并验证的能力，避免把规划中的指标写成既成事实。

## 来源与许可说明

本项目基于 [lei44196/Aagent_zhisaotong](https://github.com/lei44196/Aagent_zhisaotong) 改造，保留原始 Git 历史以便审计与归因。原仓库 README 声明采用 MIT License，但源码快照中未包含对应 `LICENSE` 文件；在上游补齐许可文件或获得作者确认前，请勿将本项目用于商业分发。
