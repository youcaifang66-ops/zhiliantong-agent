"""
事实提取模块：用 LLM 从对话消息中提取关键事实
支持 Hash 精确去重 + Embedding 语义去重
"""
import json
import hashlib
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from model.factory import chat_model
from utils.logger_handler import logger

EXTRACT_PROMPT = """你是一个信息提取助手。请从以下对话中提取用户的关键事实和偏好。

【提取要求】
1. 只提取确定性的事实，不要推断或猜测
2. 提取的内容包括：
   - 用户的个人信息（姓名、年龄、地址等）
   - 用户的环境信息（房屋面积、户型、宠物等）
   - 用户的产品偏好（预算、功能需求、品牌偏好等）
   - 用户的购买意向（已下单、犹豫中、对比中等）
   - 用户关注的问题（售后、耗材、噪音等）
3. 每条事实用一句话描述，控制在 30 字以内
4. 以 JSON 数组格式返回，例如：["用户家120平", "用户养猫", "用户偏好静音模式"]
5. 如果没有可提取的事实，返回空数组 []

【对话内容】
{conversation}

请直接返回 JSON 数组，不要加任何前缀或解释：
"""

# 构建 LangChain 链：prompt → LLM → 字符串输出
extract_chain = (
    ChatPromptTemplate.from_template(EXTRACT_PROMPT)
    | chat_model
    | StrOutputParser()
)


def _hash_fact(fact: str) -> str:
    """对事实文本做 MD5 哈希，用于精确去重"""
    return hashlib.md5(fact.strip().encode("utf-8")).hexdigest()


def dedup_facts(facts: list, existing_hashes: set = None) -> list:
    """
    Hash 精确去重：去除完全相同的事实

    参数:
        facts: 新提取的事实列表
        existing_hashes: 已存在事实的哈希集合（从 ChromaDB 加载），为 None 时仅做内部去重

    返回:
        去重后的事实列表
    """
    if existing_hashes is None:
        existing_hashes = set()

    seen = set()
    unique_facts = []
    for fact in facts:
        fact_hash = _hash_fact(fact)
        if fact_hash not in existing_hashes and fact_hash not in seen:
            seen.add(fact_hash)
            unique_facts.append(fact)

    dedup_count = len(facts) - len(unique_facts)
    if dedup_count > 0:
        logger.info(f"[事实去重] Hash去重：{len(facts)}条 → {len(unique_facts)}条，去除{dedup_count}条重复")

    return unique_facts


def extract_facts(messages: list) -> list:
    """
    从对话消息中提取关键事实

    参数:
        messages: 消息列表 [{"role": "user", "content": "..."}, ...]

    返回:
        事实列表 ["用户家120平", "用户养猫", ...] 或空列表
    """
    # 1. 将消息列表拼接成对话文本
    conversation = "\n".join([
        f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}"
        for m in messages
    ])

    if not conversation.strip():
        return []

    try:
        # 2. 调用 LLM 提取事实
        result = extract_chain.invoke({"conversation": conversation})

        # 3. 解析 JSON
        facts = json.loads(result)

        # 4. 校验
        if not isinstance(facts, list):
            logger.warning(f"[事实提取] LLM 返回的不是列表：{result}")
            return []

        facts = [f for f in facts if isinstance(f, str) and f.strip()]
        facts = [f[:50] for f in facts]

        # 5. Hash 精确去重
        facts = dedup_facts(facts)

        return facts

    except json.JSONDecodeError as e:
        logger.warning(f"[事实提取] JSON 解析失败：{e}，LLM 返回：{result}")
        return []
    except Exception as e:
        logger.error(f"[事实提取] 提取失败：{e}")
        return []