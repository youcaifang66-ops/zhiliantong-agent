"""
摘要通道模块：用 LLM 将对话压缩为摘要，持久化存入 JSON 文件
"""
import json
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from model.factory import chat_model
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

MEMORY_FILE = "data/memory_store.json"

SUMMARY_PROMPT = """请将以下对话压缩为一段简洁的记忆摘要，要求：
1. 保留关键事实（用户提到的个人信息、偏好、需求等）
2. 保留对话的主要脉络（聊了什么话题、达成了什么结论）
3. 控制在 200 字以内
4. 用第三人称描述，例如"用户提到..."

对话内容：
{conversation}

请直接输出摘要，不要加任何前缀：
"""

summary_chain = (
    ChatPromptTemplate.from_template(SUMMARY_PROMPT)
    | chat_model
    | StrOutputParser()
)


class SummaryMemory:
    def __init__(self):
        self.file_path = get_abs_path(MEMORY_FILE)
        self._ensure_file()

    def _ensure_file(self):
        """确保 JSON 文件存在"""
        dir_path = os.path.dirname(self.file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def save_summary(self, user_id: str, summary: str):
        """
        将该用户的对话摘要存入 JSON 文件

        参数:
            user_id: 用户唯一标识
            summary: 压缩后的摘要文本
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if user_id not in data:
            data[user_id] = {"summaries": [], "summary": ""}

        data[user_id]["summaries"].append(summary)
        # 合并所有历史摘要
        data[user_id]["summary"] = "\n".join(data[user_id]["summaries"])

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[摘要记忆] 用户 {user_id} 已更新摘要")

    def get_user_summary(self, user_id: str) -> str:
        """
        读取该用户的历史摘要

        参数:
            user_id: 用户唯一标识

        返回:
            摘要文本，如果没有则返回空字符串
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get(user_id, {}).get("summary", "")


def compress_to_summary(messages: list) -> str:
    """
    用 LLM 将对话消息压缩为摘要

    参数:
        messages: 消息列表 [{"role": "user", "content": "..."}, ...]

    返回:
        摘要文本
    """
    conversation = "\n".join([
        f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}"
        for m in messages
    ])

    if not conversation.strip():
        return ""

    try:
        summary = summary_chain.invoke({"conversation": conversation})
        return summary.strip()
    except Exception as e:
        logger.error(f"[摘要压缩] 失败：{e}")
        return ""