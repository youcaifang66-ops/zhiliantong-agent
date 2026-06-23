"""
中间件模块：
  1. monitor_tool — 工具调用监控与日志记录
  2. log_before_model — 模型调用前日志记录
  3. intent_prompt_switch — 基于意图分类器的动态提示词切换（兼容 report 标记）
"""
from typing import Callable
from utils.prompt_loader import load_system_prompts, load_prompt_by_file
from agent.intent_classifier import classify_intent, get_prompt_file
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """工具执行的监控"""
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        # fill_context_for_report 调用后标记 report，供 intent_prompt_switch 读取
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        state: AgentState,
        runtime: Runtime,
):
    """在模型执行前输出日志"""
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None


@dynamic_prompt
def intent_prompt_switch(request: ModelRequest):
    """
    基于意图分类器的动态提示词切换

    优先级：
    1. context["report"] == True → 报告生成提示词（LLM 运行时调用 fill_context_for_report 触发）
    2. context["intent"] → 对应意图的提示词（预判注入）
    3. 对最后一条用户消息做分类（兜底）
    """
    # 优先检查 report 标记（LLM 运行时触发的场景切换）
    if request.runtime.context.get("report", False):
        logger.info("[意图切换] report=True → report_prompt.txt（运行时切换）")
        return load_prompt_by_file("report_prompt.txt")

    # 读取预判的 intent
    intent = request.runtime.context.get("intent", None)

    # 如果没有预判过，对最后一条用户消息做分类
    if intent is None:
        messages = request.state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                intent = classify_intent(msg.content)
                request.runtime.context["intent"] = intent
                break

    if intent is None:
        return load_system_prompts()

    prompt_file = get_prompt_file(intent)
    logger.info(f"[意图切换] intent={intent} → prompt={prompt_file}")

    prompt = load_prompt_by_file(prompt_file)

    # 将记忆上下文追加到意图 prompt 末尾（解决记忆被覆盖的问题）
    memory_context = request.runtime.context.get("memory_context", "")
    if memory_context:
        prompt += f"\n\n{memory_context}"

    return prompt
