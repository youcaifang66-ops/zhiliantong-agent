from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, intent_prompt_switch
from agent.intent_classifier import classify_intent
from agent.memory_manager import MemoryManager
from utils.logger_handler import logger


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, intent_prompt_switch],
        )
        self.memory_manager = MemoryManager()

    def execute_stream(self, query: str, user_id: str = "default", history: list = None):
        if history is None:
            history = []

        # 1. 追加用户消息到短期记忆
        history.append({"role": "user", "content": query})

        # 2. 滑动窗口：超出时先双通道处理再丢弃
        history = self.memory_manager.trim_window(history, user_id)

        # 3. 意图预判（在 Agent 执行前完成）
        intent = classify_intent(query)
        logger.info(f"[ReactAgent] 用户意图：{intent}")

        # 4. 构建记忆上下文（不含基础提示词，供中间件注入到任意意图 prompt）
        memory_context = self.memory_manager.build_memory_context(user_id, query)

        # 5. 构造完整消息列表（system + 短期记忆）
        input_dict = {
            "messages": [{"role": "system", "content": ""}] + history,
        }

        # 6. 流式调用 Agent，将预判的 intent 和记忆上下文注入 context
        full_response = ""
        for chunk in self.agent.stream(
            input_dict,
            stream_mode="values",
            context={"intent": intent, "memory_context": memory_context},
        ):
            latest_message = chunk["messages"][-1]
            if isinstance(latest_message, AIMessage) and latest_message.content:
                content = latest_message.content.strip() + "\n"
                full_response += content
                yield content

        # 7. 追加助手回复到短期记忆
        if full_response.strip():
            history.append({"role": "assistant", "content": full_response.strip()})


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
