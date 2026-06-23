"""
意图分类器：基于规则 + 关键词的轻量意图预判

在主 Agent 执行前完成意图识别，将 intent 注入 context，
中间件根据 intent 动态切换提示词与工具集。

支持的意图：
  - 咨询问答 (qa)：一般性产品问题、使用建议
  - 故障排查 (troubleshoot)：错误码、故障现象、异常处理
  - 维护保养 (maintenance)：耗材更换、清洁保养、定期维护
  - 报告生成 (report)：使用报告、数据统计、使用记录查询
  - 选购推荐 (purchase)：选购建议、户型匹配、功能对比
"""
import re
from utils.logger_handler import logger

# ============================================================
# 意图规则定义：关键词 + 正则 + 权重
# ============================================================
INTENT_RULES = {
    "troubleshoot": {
        "keywords": [
            "故障", "错误", "报错", "异常", "坏了", "失灵", "不转", "不动",
            "无法", "不能", "不工作", "不启动", "不开机", "断电", "死机",
            "卡住", "卡死", "缠绕", "迷路", "回不去", "找不到", "丢失",
            "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9",
            "错误码", "error", "fault",
            "不充电", "充不进电", "电量", "电池",
            "不吸", "吸力", "没吸力", "吸力弱",
            "漏水", "渗水", "水箱",
            "噪音大", "声音大", "异响", "响声",
            "掉线", "断连", "连不上", "离线",
        ],
        "patterns": [
            r"显示.*码",
            r"出现.*错误",
            r"提示.*故障",
            r"一直.*闪",
            r"怎么.*修",
            r"怎么.*解决",
        ],
    },
    "maintenance": {
        "keywords": [
            "保养", "维护", "清洁", "清洗", "清理", "除尘",
            "滤网", "边刷", "主刷", "滚刷", "万向轮", "拖布",
            "更换", "替换", "多久换", "什么时候换", "耗材",
            "寿命", "使用周期", "定期",
            "保养方法", "保养建议", "维护建议",
        ],
        "patterns": [
            r"多久.*换",
            r"多久.*清洁",
            r"多久.*保养",
            r"多久.*清理",
            r"怎么.*保养",
            r"怎么.*维护",
        ],
    },
    "report": {
        "keywords": [
            "报告", "使用报告", "使用记录", "使用数据",
            "月度", "统计", "数据", "报表",
            "生成报告", "导出报告", "查看报告",
            "清洁记录", "清扫记录",
        ],
        "patterns": [
            r"生成.*报告",
            r"查看.*记录",
            r"导出.*数据",
            r"\d+月.*报告",
        ],
    },
    "purchase": {
        "keywords": [
            "选购", "推荐", "购买", "买", "入手", "哪款", "哪个",
            "型号", "款式", "品牌", "对比", "比较",
            "适合", "适配", "匹配",
            "预算", "价位", "价格", "多少钱",
            "户型", "面积", "平", "大户型", "小户型",
            "养猫", "养狗", "宠物", "毛发",
            "功能", "配置", "参数",
        ],
        "patterns": [
            r"推荐.*款",
            r"哪款.*好",
            r"适合.*用",
            r"多少平.*用",
            r"养.*推荐",
        ],
    },
}

# 意图优先级：当多个意图匹配时，按此顺序选择
INTENT_PRIORITY = ["troubleshoot", "report", "maintenance", "purchase", "qa"]

# 意图到提示词文件的映射
INTENT_PROMPT_MAP = {
    "qa": "main_prompt.txt",
    "troubleshoot": "troubleshoot_prompt.txt",
    "maintenance": "maintenance_prompt.txt",
    "report": "report_prompt.txt",
    "purchase": "purchase_prompt.txt",
}


def classify_intent(query: str) -> str:
    """
    基于规则 + 关键词的轻量意图分类

    参数:
        query: 用户输入文本

    返回:
        意图标签：qa / troubleshoot / maintenance / report / purchase
    """
    query_lower = query.lower()
    scores = {intent: 0 for intent in INTENT_RULES}

    for intent, rules in INTENT_RULES.items():
        # 关键词匹配
        for keyword in rules["keywords"]:
            if keyword.lower() in query_lower:
                scores[intent] += 1

        # 正则匹配（权重更高）
        for pattern in rules["patterns"]:
            if re.search(pattern, query):
                scores[intent] += 2  # 正则匹配权重 ×2

    # 按分数降序排列
    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 最高分 > 0 且高于第二名，直接返回
    if sorted_intents[0][1] > 0:
        top_intent = sorted_intents[0][0]

        # 多意图并列时按优先级选择
        if sorted_intents[0][1] == sorted_intents[1][1] and sorted_intents[1][1] > 0:
            for intent in INTENT_PRIORITY:
                if scores[intent] == sorted_intents[0][1]:
                    top_intent = intent
                    break

        logger.info(f"[意图识别] query=\"{query}\" → {top_intent} (score={scores[top_intent]})")
        return top_intent

    # 无匹配，默认咨询问答
    logger.info(f"[意图识别] query=\"{query}\" → qa (默认)")
    return "qa"


def get_prompt_file(intent: str) -> str:
    """根据意图返回对应的提示词文件名"""
    return INTENT_PROMPT_MAP.get(intent, "main_prompt.txt")


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    test_queries = [
        "扫地机显示E4错误码怎么办",
        "边刷多久换一次",
        "帮我生成3月使用报告",
        "120平养猫适合哪款扫地机",
        "扫地机怎么连接WiFi",
        "噪音很大怎么回事",
        "滤网怎么清洗",
        "推荐一款性价比高的",
    ]
    for q in test_queries:
        intent = classify_intent(q)
        print(f"  {q:30s} → {intent}")
