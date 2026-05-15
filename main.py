import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from agent.react_agent import ReactAgent

app = FastAPI(title="智扫通智能客服")

# 请求体定义
class ChatRequest(BaseModel):
    query: str


# 全局初始化 Agent（只初始化一次）
agent = ReactAgent()


@app.post("/chat")
async def chat(request: ChatRequest):
    """聊天接口，流式返回"""

    async def generate():
        for chunk in agent.execute_stream(request.query):
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
