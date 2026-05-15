"""
智扫通智能客服 - FastAPI 服务入口

【FastAPI 核心概念学习】
- FastAPI()：创建应用实例，相当于 Django 的  django-admin startproject
- @app.get/post()：定义路由，相当于 Django 的 urls.py + views.py 合体
- pydantic.BaseModel：请求体校验，相当于 Django REST Framework 的 Serializer
- StreamingResponse：流式响应，逐块返回数据（聊天场景必备）
- uvicorn：运行 FastAPI 的服务器，相当于 Django 的 runserver
"""

import json
import sys
import os
import traceback  # 打印完整错误堆栈，方便调试

# 将项目根目录加入 Python 搜索路径，这样才能 import agent 下的模块
# 【FastAPI 要点】FastAPI 不会自动设置项目路径，需要手动添加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI  # 【FastAPI】核心类，创建应用
from fastapi.responses import StreamingResponse, HTMLResponse  # 【FastAPI】StreamingResponse:流式响应, HTMLResponse:返回HTML页面
from fastapi.middleware.cors import CORSMiddleware  # 【FastAPI】跨域中间件，让前端能调用接口
from fastapi.staticfiles import StaticFiles  # 【FastAPI】挂载静态文件目录（CSS/JS/图片等）
from pydantic import BaseModel  # 【FastAPI/Pydantic】数据校验模型，相当于 Django 的 forms/DRF Serializers
from agent.react_agent import ReactAgent  # 复用你已有的 Agent 代码，一行不改

# ============================================================
# 【FastAPI】创建应用实例
# ============================================================
# 相当于 Django 项目中的 FastAPI 替代了 django-admin startproject 创建的项目实例
# title 会在 /docs 页面显示为标题
app = FastAPI(title="智扫通智能客服", description="基于 LLM+RAG+Agent 的扫地机器人智能问答系统")

# ============================================================
# 【FastAPI】配置跨域
# ============================================================
# 前端页面（HTML）和后端服务（FastAPI）通常是不同端口/域名
# 跨域配置就是允许前端页面调用后端接口，相当于 Django 的 django-cors-headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允许所有来源访问（生产环境应改为具体域名）
    allow_methods=["*"],       # 允许所有 HTTP 方法（GET/POST/PUT/DELETE 等）
    allow_headers=["*"],       # 允许所有请求头
)

# ============================================================
# 【FastAPI】挂载静态文件和提供前端页面
# ============================================================
# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 【FastAPI】挂载静态文件目录
# 访问 /static/xxx 时，自动从 static/ 目录下找文件
# 相当于 Django 中 settings.py 的 STATIC_URL + STATICFILES_DIRS
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# 【FastAPI】根路径返回前端页面
# @app.get("/") 当用户访问 http://localhost:8000/ 时触发
@app.get("/")
async def index():
    """返回聊天前端页面"""
    html_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # HTMLResponse 告诉浏览器这是一个 HTML 页面
    return HTMLResponse(content=html_content)

# ============================================================
# 【FastAPI】全局初始化 Agent（只初始化一次，所有请求复用）
# ============================================================
# 注意：这里在模块加载时就会初始化，启动后所有请求共享这一个 agent 实例
agent = ReactAgent()


# ============================================================
# 【FastAPI/Pydantic】定义请求体模型
# ============================================================
# 相当于 Django 中 forms.Form 或 DRF 中 serializers.Serializer
# FastAPI 会自动根据这个模型校验请求体，校验失败自动返回 422 错误
class ChatRequest(BaseModel):
    query: str  # 用户输入的问题


# ============================================================
# 【FastAPI】路由定义：聊天接口（流式响应）
# ============================================================
# @app.post("/chat") 相当于 Django 中 path("chat/", views.chat) + POST 限制
# async def：异步函数，不阻塞服务器处理其他请求
@app.post("/chat")
async def chat(request: ChatRequest):
    """
    聊天接口
    - 接收用户问题（POST 请求的 JSON body）
    - 返回流式响应（一段一段返回文字，类似 ChatGPT 的打字效果）

    技术原理：
    普通接口是等全部结果算完一次性返回
    流式接口是算一点返回一点，用户体验更好
    """

    # 【FastAPI】同步生成器函数
    # 由于 agent.execute_stream() 是同步的，这里也用同步生成器
    # StreamingResponse 支持同步和异步两种生成器
    def generate():
        try:
            # agent.execute_stream 返回一个生成器，逐块吐出文字
            for chunk in agent.execute_stream(request.query):
                # SSE(Server-Sent Events) 格式：data: 内容\n\n
                # 前端通过 EventSource 或 fetch 读取这个格式
                yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            # 把完整错误堆栈打印到控制台，方便排查
            traceback.print_exc()
            # 同时把错误信息返回给前端
            yield f"data: {json.dumps({'error': f'[{type(e).__name__}] {str(e)}'}, ensure_ascii=False)}\n\n"

    # 【FastAPI】返回流式响应
    # media_type="text/event-stream" 告诉浏览器这是 SSE 流
    return StreamingResponse(generate(), media_type="text/event-stream")


# ============================================================
# 【FastAPI】路由定义：非流式接口（一次性返回结果）
# ============================================================
# 相比流式接口，这个更简单：等 Agent 全部算完，一次性返回结果
# 适合简单场景或对实时性要求不高的调用方
@app.post("/chat/sync")
async def chat_sync(request: ChatRequest):
    """
    同步聊天接口（非流式）
    - 和 /chat 功能一样，但等全部结果算完才返回
    """
    full_response = ""
    for chunk in agent.execute_stream(request.query):
        full_response += chunk

    return {"query": request.query, "response": full_response.strip()}


# ============================================================
# 【FastAPI】路由定义：健康检查接口
# ============================================================
# GET 请求，无参数，返回服务状态
# 相当于 Django 中一个简单的健康检查视图
@app.get("/health")
async def health():
    """健康检查接口，用于判断服务是否正常运行"""
    return {
        "status": "ok",
        "service": "智扫通智能客服",
        "agent_ready": agent is not None,
    }


# ============================================================
# 【FastAPI】直接运行入口
# ===========================================================
# 支持两种启动方式：
# 方式1（推荐）：uvicorn fastapi_main:app --reload --port 8000
# 方式2：python fastapi_main.py
if __name__ == "__main__":
    import uvicorn
    # 【FastAPI】启动服务
    # host="0.0.0.0" 允许局域网内其他设备访问
    # port=8000 端口号
    # reload=True 代码修改后自动重启（开发时用）
    uvicorn.run("fastapi_main:app", host="0.0.0.0", port=8001, reload=True)