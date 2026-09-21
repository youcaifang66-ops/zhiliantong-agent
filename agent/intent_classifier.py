"""健身场景的轻量意图路由。

先用可解释规则预判意图，再由 Agent 中间件切换场景提示词。规则路由
不依赖模型，适合作为低成本、可回归的第一层分流。
"""
import re

from utils.logger_handler import logger


INTENT_RULES = {
    "plan": {
        "keywords": ["计划", "安排", "组合", "增肌", "减脂", "塑形", "一周", "训练几天", "训练方案", "器械"],
        "patterns": [r"每周.*练", r"帮我.*计划", r"怎么.*安排", r"制定.*训练"],
    },
    "record": {
        "keywords": ["记录", "完成", "打卡", "训练日志"],
        "patterns": [r"记录.*组", r"今天.*练了", r"完成.*次", r"打卡.*训练"],
    },
    "adjust": {
        "keywords": ["调整", "太轻", "太重", "疼痛", "疲劳", "平台期", "加重量", "减量", "恢复", "内扣", "代偿"],
        "patterns": [r"练完.*疼", r"重量.*调整", r"多久.*加", r"没有.*进步"],
    },
    "report": {
        "keywords": ["报告", "复盘", "统计", "趋势", "月报", "周报", "训练记录"],
        "patterns": [r"生成.*报告", r"查看.*记录", r"分析.*训练", r"\d+月.*报告"],
    },
}

INTENT_PRIORITY = ["adjust", "report", "record", "plan", "qa"]

INTENT_PROMPT_MAP = {
    "qa": "main_prompt.txt",
    "plan": "plan_prompt.txt",
    "record": "record_prompt.txt",
    "adjust": "adjust_prompt.txt",
    "report": "report_prompt.txt",
}


def classify_intent(query: str) -> str:
    """返回 qa / plan / record / adjust / report 中的一个标签。"""
    scores = {intent: 0 for intent in INTENT_RULES}
    query_lower = query.lower()

    for intent, rules in INTENT_RULES.items():
        for keyword in rules["keywords"]:
            if keyword.lower() in query_lower:
                scores[intent] += 1
        for pattern in rules["patterns"]:
            if re.search(pattern, query, flags=re.IGNORECASE):
                scores[intent] += 2

    best_score = max(scores.values(), default=0)
    if best_score == 0:
        logger.info('[意图识别] query="%s" -> qa (默认)', query)
        return "qa"

    for intent in INTENT_PRIORITY:
        if scores.get(intent) == best_score:
            logger.info('[意图识别] query="%s" -> %s (score=%s)', query, intent, best_score)
            return intent
    return "qa"


def get_prompt_file(intent: str) -> str:
    return INTENT_PROMPT_MAP.get(intent, "main_prompt.txt")


if __name__ == "__main__":
    for sample in ["帮我制定一周增肌计划", "记录深蹲4组", "膝盖疼怎么调整", "生成本月训练报告"]:
        print(sample, "->", classify_intent(sample))
