"""
向量通道模块：将提取的事实向量化存入 ChromaDB，并提供检索能力
支持 Embedding 语义去重 + TTL 过期衰减 + Importance 重要性权重
"""
import time
from langchain_chroma import Chroma
from langchain_core.documents import Document
from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.logger_handler import logger

# 半衰期：记忆权重衰减到 0.5 所需时间（秒），7天
DEFAULT_HALF_LIFE = 7 * 24 * 3600
# Importance 衰减速率：每过一个半衰期，权重衰减为原来的 0.5
DECAY_RATE = 0.5
# 权重阈值：低于此值的记忆视为过期，检索时丢弃
WEIGHT_THRESHOLD = 0.1
# 语义去重阈值：余弦相似度 > 此值视为重复
SIMILARITY_THRESHOLD = 0.85


class VectorMemory:
    def __init__(self):
        """复用已有 ChromaDB，使用独立的 collection 存储长期记忆"""
        self.store = Chroma(
            collection_name="long_term_memory",
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )

    def _get_existing_hashes(self, user_id: str) -> set:
        """获取该用户所有已存事实的 Hash，用于精确去重"""
        try:
            results = self.store.get(
                filter={"user_id": user_id},
                include=["metadatas"],
            )
            if results and results["metadatas"]:
                return {m.get("fact_hash", "") for m in results["metadatas"] if m.get("fact_hash")}
        except Exception as e:
            logger.warning(f"[向量记忆] 获取已有Hash失败：{e}")
        return set()

    def _embedding_dedup(self, user_id: str, facts: list) -> list:
        """
        Embedding 语义去重：用余弦相似度判断新事实与已有事实是否语义重复

        参数:
            user_id: 用户ID
            facts: 待存入的事实列表

        返回:
            去重后的事实列表
        """
        if not facts:
            return []

        existing_hashes = self._get_existing_hashes(user_id)
        if not existing_hashes:
            return facts

        try:
            # 获取该用户所有已有事实的文本
            results = self.store.get(
                filter={"user_id": user_id},
                include=["documents", "embeddings"],
            )
            if not results or not results["documents"]:
                return facts

            existing_docs = results["documents"]
            existing_embeddings = results["embeddings"]

            # 对新事实做 embedding
            new_embeddings = embed_model.embed_documents(facts)

            unique_facts = []
            for i, fact in enumerate(facts):
                is_duplicate = False
                new_emb = new_embeddings[i]

                for existing_emb in existing_embeddings:
                    # 计算余弦相似度
                    similarity = self._cosine_similarity(new_emb, existing_emb)
                    if similarity > SIMILARITY_THRESHOLD:
                        is_duplicate = True
                        logger.info(f"[语义去重] \"{fact}\" 与已有记忆相似度 {similarity:.2f}，跳过")
                        break

                if not is_duplicate:
                    unique_facts.append(fact)

            dedup_count = len(facts) - len(unique_facts)
            if dedup_count > 0:
                logger.info(f"[语义去重] Embedding去重：{len(facts)}条 → {len(unique_facts)}条")

            return unique_facts

        except Exception as e:
            logger.warning(f"[语义去重] 去重失败，直接存入：{e}")
            return facts

    @staticmethod
    def _cosine_similarity(vec_a: list, vec_b: list) -> float:
        """计算两个向量的余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def save_facts(self, user_id: str, facts: list):
        """
        将事实列表向量化存入 ChromaDB（含双重去重）

        参数:
            user_id: 用户唯一标识
            facts: 事实列表 ["用户家120平", "用户养猫", ...]
        """
        if not facts:
            return

        # 1. Hash 精确去重（在 memory_extractor 中已完成）
        # 2. Embedding 语义去重
        facts = self._embedding_dedup(user_id, facts)

        if not facts:
            logger.info(f"[向量记忆] 用户 {user_id} 所有事实均重复，跳过存入")
            return

        # 3. 构建带半衰期和 Importance 的 Document
        now = time.time()
        documents = [
            Document(
                page_content=fact,
                metadata={
                    "user_id": user_id,
                    "fact_hash": self._hash_fact(fact),
                    "created_at": now,
                    "half_life": DEFAULT_HALF_LIFE,
                    "importance": 1.0,
                },
            )
            for fact in facts
        ]

        self.store.add_documents(documents)
        logger.info(f"[向量记忆] 用户 {user_id} 已存入 {len(facts)} 条事实")

    def _calculate_weight(self, doc: Document) -> float:
        """
        计算记忆权重：基于半衰期指数衰减 + Importance

        权重公式：weight = importance * (DECAY_RATE ^ elapsed_periods)
        其中 elapsed_periods = (now - created_at) / half_life
        """
        metadata = doc.metadata
        importance = metadata.get("importance", 1.0)
        created_at = metadata.get("created_at", time.time())
        half_life = metadata.get("half_life", DEFAULT_HALF_LIFE)

        elapsed = time.time() - created_at
        periods = elapsed / half_life if half_life > 0 else 0
        weight = importance * (DECAY_RATE ** periods)

        return round(weight, 4)

    def search_memory(self, user_id: str, query: str, top_k: int = 3) -> list:
        """
        检索该用户与当前问题相关的事实（权重衰减排序 + 阈值过滤）

        参数:
            user_id: 用户ID，用于过滤
            query: 当前用户提问，用于相似度检索
            top_k: 返回最相关的 K 条

        返回:
            事实文本列表 ["用户家120平", "用户养猫", ...]
        """
        try:
            # 多检索一些，过滤后取 top_k
            results = self.store.similarity_search(
                query=query,
                k=top_k * 3,
                filter={"user_id": user_id},
            )

            # 权重衰减排序 + 阈值过滤（不再硬截断 TTL）
            weighted_results = []
            for doc in results:
                weight = self._calculate_weight(doc)

                # 权重低于阈值的记忆视为过期，丢弃
                if weight < WEIGHT_THRESHOLD:
                    logger.info(f"[权重衰减] 丢弃低权重记忆：\"{doc.page_content[:20]}...\" (权重={weight})")
                    continue

                weighted_results.append((doc.page_content, weight))

            # 按权重降序排列，取 top_k
            weighted_results.sort(key=lambda x: x[1], reverse=True)
            final_results = [text for text, _ in weighted_results[:top_k]]

            if final_results:
                logger.info(f"[向量记忆] 用户 {user_id} 检索到 {len(final_results)} 条记忆")

            return final_results

        except Exception as e:
            logger.error(f"[向量记忆] 检索失败：{e}")
            return []

    def cleanup_expired(self, user_id: str):
        """清理低权重记忆（可定期调用）"""
        try:
            results = self.store.get(
                filter={"user_id": user_id},
                include=["documents", "metadatas", "ids"],
            )
            if not results or not results["ids"]:
                return

            expired_ids = []
            for i, metadata in enumerate(results["metadatas"]):
                # 用临时 Document 计算权重
                doc = Document(page_content=results["documents"][i], metadata=metadata)
                weight = self._calculate_weight(doc)
                if weight < WEIGHT_THRESHOLD:
                    expired_ids.append(results["ids"][i])

            if expired_ids:
                self.store.delete(ids=expired_ids)
                logger.info(f"[权重清理] 用户 {user_id} 清理了 {len(expired_ids)} 条低权重记忆")

        except Exception as e:
            logger.warning(f"[权重清理] 清理失败：{e}")
