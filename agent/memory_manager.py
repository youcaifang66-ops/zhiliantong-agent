"""
记忆管理器：统一管理短期记忆（滑动窗口）和长期记忆（双通道）

核心逻辑：
  1. 每轮对话时追加消息，检查滑动窗口是否超出
  2. 超出时先对旧消息做双通道处理（向量通道+摘要通道），再丢弃
  3. 会话结束时对剩余消息补做双通道处理
"""
from agent.vector_memory import VectorMemory
from agent.summary_memory import SummaryMemory, compress_to_summary
from agent.memory_extractor import extract_facts, dedup_facts
from utils.logger_handler import logger

MAX_TURNS = 10


class MemoryManager:
    def __init__(self):
        self.vector_memory = VectorMemory()
        self.summary_memory = SummaryMemory()

    def trim_window(self, history: list, user_id: str) -> list:
        """
        滑动窗口：超出时先双通道处理，再丢弃旧消息

        参数:
            history: 当前对话消息列表
            user_id: 用户唯一标识

        返回:
            处理后的消息列表（保留最近 10 轮）
        """
        max_messages = MAX_TURNS * 2  # 10轮 = 20条

        if len(history) <= max_messages:
            return history

        # 超出部分（要丢弃的旧消息）
        overflow = history[:-max_messages]
        remaining = history[-max_messages:]

        # 丢弃前做双通道处理
        self._process_both_channels(overflow, user_id)

        logger.info(
            f"[滑动窗口] 用户 {user_id} 丢弃 {len(overflow)} 条旧消息，"
            f"保留 {len(remaining)} 条"
        )

        return remaining

    def _process_both_channels(self, messages: list, user_id: str):
        """
        双通道处理：向量通道 + 摘要通道

        参数:
            messages: 待处理的消息列表
            user_id: 用户唯一标识
        """
        # 通道1：提取事实 → Hash去重 → Embedding去重 → ChromaDB
        facts = extract_facts(messages)
        if facts:
            # 获取已有 Hash 做精确去重
            existing_hashes = self.vector_memory._get_existing_hashes(user_id)
            facts = dedup_facts(facts, existing_hashes)
            if facts:
                self.vector_memory.save_facts(user_id, facts)

        # 通道2：压缩摘要 → JSON
        summary = compress_to_summary(messages)
        if summary:
            self.summary_memory.save_summary(user_id, summary)

    def build_memory_context(self, user_id: str, query: str) -> str:
        """
        构建记忆上下文文本（不含基础提示词，供中间件注入到任意 prompt）

        参数:
            user_id: 用户唯一标识
            query: 当前用户提问

        返回:
            记忆上下文文本，如果无记忆则返回空字符串
        """
        parts = []

        # 读取摘要通道
        summary = self.summary_memory.get_user_summary(user_id)
        if summary:
            parts.append(f"## 用户对话历史\n{summary}")

        # 检索向量通道（自动带权重衰减排序 + 阈值过滤）
        facts = self.vector_memory.search_memory(user_id, query, top_k=3)
        if facts:
            facts_text = "\n".join([f"- {fact}" for fact in facts])
            parts.append(f"## 用户相关记忆\n{facts_text}")

        return "\n\n".join(parts)

    def build_system_prompt(self, user_id: str, query: str) -> str:
        """
        构建带有长期记忆的 system prompt

        参数:
            user_id: 用户唯一标识
            query: 当前用户提问

        返回:
            注入长期记忆后的 system prompt 文本
        """
        from utils.prompt_loader import load_system_prompts
        base_prompt = load_system_prompts()

        # 读取摘要通道
        summary = self.summary_memory.get_user_summary(user_id)
        if summary:
            base_prompt += f"\n\n## 用户对话历史\n{summary}"

        # 检索向量通道（自动带 TTL 过期过滤 + 权重衰减排序）
        facts = self.vector_memory.search_memory(user_id, query, top_k=3)
        if facts:
            facts_text = "\n".join([f"- {fact}" for fact in facts])
            base_prompt += f"\n\n## 用户相关记忆\n{facts_text}"

        return base_prompt

    def on_session_end(self, user_id: str, history: list):
        """
        会话结束时对剩余消息做双通道处理 + 清理过期记忆

        参数:
            user_id: 用户唯一标识
            history: 当前会话的所有消息
        """
        if not history:
            return

        logger.info(f"[会话结束] 用户 {user_id}，处理剩余 {len(history)} 条消息")
        self._process_both_channels(history, user_id)

        # 清理过期记忆
        self.vector_memory.cleanup_expired(user_id)

        history.clear()
