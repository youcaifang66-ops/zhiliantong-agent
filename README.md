# 智练通（FitFlow Agent）

面向个人训练计划、训练记录、动态调整与周期复盘的健身 Agent。项目重点解决 Agent 的可控路由、知识溯源、长期记忆、幂等工具、任务恢复和离线评测。

## 已实现能力

- **LangGraph 工作流**：意图路由 → 安全检查 → 混合检索 → 引用组装，关键节点可追踪并支持线程检查点。
- **类型化工具**：Pydantic 校验训练动作、重量、次数、时长和 RPE；SQLite 仓储通过 `request_id` 保证重复请求不重复写入。
- **三层记忆**：短期滑动窗口、长期事实、会话摘要；长期事实使用 SHA-256 精确去重、字符相似度语义去重、TTL 与 90 天指数衰减。
- **混合检索**：BM25 与离线稠密字符向量双路召回，使用 RRF 融合并返回来源、分块 ID、通道与融合分数。
- **可靠执行**：SQLite 任务检查点保存步骤、状态、负载和重试次数，工具失败最多重试 3 次，可恢复最终状态。
- **API 工程**：FastAPI 提供健康检查、训练记录、月度汇总、同步 Agent 和 SSE 轨迹接口。
- **交付保障**：uv 锁定 Python 3.12 依赖，Ruff、pytest、GitHub Actions、Dockerfile 与健康检查齐全。

## 可复现评测结果

执行：

```bash
uv sync --frozen --group dev
uv run pytest -q
uv run python eval/run_offline_eval.py
```

`eval/fitness_golden.json` 包含 50 条人工标注意图与目标来源的 Golden Query。当前提交实测结果：

| 指标 | 结果 |
|---|---:|
| 意图准确率 | 100% |
| Recall@1 | 90% |
| Recall@3 | 98% |
| Recall@5 | 100% |
| MRR | 0.9417 |
| NDCG@5 | 0.9565 |
| 离线检索 P95 | 0.264 ms |

这些数字来自仓库内固定数据和脚本，不包含模型生成质量。生产链路可替换为 DashScope Embedding；离线评测使用确定性稠密字符向量，保证 CI 不依赖外部模型服务。

## 快速开始

```bash
uv sync --frozen --group dev
uv run uvicorn api_app:app --host 0.0.0.0 --port 8000
```

接口文档：`http://localhost:8000/docs`。

```bash
curl -X POST http://localhost:8000/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"user_id":"1001","query":"新手每周练几次"}'
```

## 文档

- [系统架构](docs/ARCHITECTURE.md)
- [十次迭代路线](docs/ROADMAP.md)
- [面试深挖与责任边界](docs/INTERVIEW.md)
- [评测结果](eval/latest_metrics.json)

## 来源与许可

本项目基于 [lei44196/Aagent_zhisaotong](https://github.com/lei44196/Aagent_zhisaotong) 改造，并通过 GitHub Fork 保留上游关系。上游 README 声明采用 MIT License，但源码快照未包含 `LICENSE`；在作者补齐许可证或另行授权前，请勿用于商业分发。
