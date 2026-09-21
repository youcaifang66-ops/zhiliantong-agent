"""智练通 Agent 的领域工具。"""
import csv
import os
import random

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


rag = RagSummarizeService()
user_ids = [str(value) for value in range(1001, 1011)]
month_arr = [f"2026-{month:02d}" for month in range(1, 13)]
external_data: dict[str, dict[str, dict[str, str]]] = {}


@tool(description="从健身知识库检索动作要领、训练原则和安全注意事项")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，用于评估户外训练条件")
def get_weather(city: str) -> str:
    return f"{city}当前晴，26摄氏度，湿度50%，AQI 21，未来6小时降雨概率较低"


@tool(description="获取当前用户所在城市")
def get_user_location() -> str:
    return random.choice(["北京", "上海", "杭州"])


@tool(description="获取当前用户ID")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，格式为YYYY-MM")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data() -> None:
    """按 user_id 和 month 加载演示训练记录。"""
    if external_data:
        return
    path = get_abs_path(agent_conf["external_data_path"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"训练记录文件 {path} 不存在")

    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            user_id = row.pop("user_id")
            month = row.pop("month")
            external_data.setdefault(user_id, {})[month] = row


@tool(description="获取指定用户在指定月份的训练汇总记录；未找到时返回空字符串")
def fetch_training_history(user_id: str, month: str) -> str:
    generate_external_data()
    try:
        return str(external_data[user_id][month])
    except KeyError:
        logger.warning("[fetch_training_history] 未找到用户 %s 在 %s 的训练记录", user_id, month)
        return ""


@tool(description="标记训练报告场景，使中间件切换到报告生成提示词")
def fill_context_for_training_report() -> str:
    return "training_report_context_ready"
