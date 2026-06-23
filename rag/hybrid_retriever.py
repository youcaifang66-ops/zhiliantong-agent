"""
混合检索模块：BM25 关键词检索 + Embedding 向量检索 + RRF 融合排序

设计思路：
  1. BM25 通道：捕获精确关键词匹配（如 "E4 错误码" 精确命中故障排除文档）
  2. Embedding 通道：捕获语义相关（如 "扫地机卡住" → "缠绕故障"）
  3. RRF 融合：将两个通道的排序结果合并，消除单一通道的偏差
"""
import jieba
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_core.documents import Document
from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.logger_handler import logger

# RRF 常数：控制排名靠后的文档权重衰减速度，常用值 60
RRF_K = 60


class HybridRetriever:
    """
    混合检索器：BM25 + Embedding + RRF 融合

    使用方式：
        retriever = HybridRetriever()
        results = retriever.search("E4错误码怎么办", top_k=5)
    """

    def __init__(self):
        # 向量存储（复用已有 ChromaDB）
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )
        self.k = chroma_conf["k"]

        # BM25 索引（延迟构建）
        self._bm25_docs = []
        self._bm25_corpus = []
        self._bm25_index = None
        self._bm25_doc_count = 0  # 记录索引构建时的文档数
        self._build_bm25_index()

    def _build_bm25_index(self):
        """
        从 ChromaDB 加载所有文档，构建 BM25 索引

        流程：ChromaDB 全量取出 → jieba 分词 → 构建 BM25 倒排索引
        """
        try:
            results = self.vector_store.get(include=["documents", "metadatas"])
            if not results or not results["documents"]:
                logger.warning("[BM25] ChromaDB 中没有文档，BM25 索引为空")
                return

            self._bm25_docs = []
            self._bm25_corpus = []

            for doc_text, metadata in zip(results["documents"], results["metadatas"]):
                # jieba 分词，BM25 需要 token 列表
                tokens = list(jieba.cut_for_search(doc_text))
                self._bm25_docs.append(Document(page_content=doc_text, metadata=metadata))
                self._bm25_corpus.append(tokens)

            self._bm25_index = BM25Okapi(self._bm25_corpus)
            self._bm25_doc_count = len(self._bm25_docs)
            logger.info(f"[BM25] 索引构建完成，共 {len(self._bm25_docs)} 条文档")

        except Exception as e:
            logger.error(f"[BM25] 索引构建失败：{e}")

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """
        BM25 关键词检索

        返回: [(Document, bm25_score), ...] 按分数降序
        """
        if self._bm25_index is None or not self._bm25_docs:
            return []

        tokens = list(jieba.cut_for_search(query))
        scores = self._bm25_index.get_scores(tokens)

        # 取 top_k 并附带分数
        scored_docs = list(zip(self._bm25_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]

    def _embedding_search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """
        Embedding 向量检索（返回带分数的结果）

        返回: [(Document, similarity_score), ...] 按分数降序
        """
        try:
            # 用 similarity_search_with_relevance_scores 获取带分数的结果
            results = self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=top_k,
            )
            return [(doc, score) for doc, score in results]

        except Exception as e:
            logger.warning(f"[Embedding] 检索失败：{e}")
            return []

    @staticmethod
    def _rrf_fusion(
        bm25_results: list[tuple[Document, float]],
        embedding_results: list[tuple[Document, float]],
        k: int = RRF_K,
    ) -> list[Document]:
        """
        RRF（Reciprocal Rank Fusion）融合排序

        公式：RRF_score(d) = Σ 1 / (k + rank_i(d))
        其中 rank_i(d) 是文档 d 在第 i 个检索通道中的排名（从 1 开始）

        参数:
            bm25_results: BM25 检索结果 [(Document, score), ...]
            embedding_results: Embedding 检索结果 [(Document, score), ...]
            k: RRF 常数，控制排名靠后文档的权重衰减，常用值 60

        返回:
            融合排序后的 Document 列表
        """
        rrf_scores = {}

        # BM25 通道贡献
        for rank, (doc, _) in enumerate(bm25_results, start=1):
            doc_key = doc.page_content
            if doc_key not in rrf_scores:
                rrf_scores[doc_key] = {"doc": doc, "score": 0.0}
            rrf_scores[doc_key]["score"] += 1.0 / (k + rank)

        # Embedding 通道贡献
        for rank, (doc, _) in enumerate(embedding_results, start=1):
            doc_key = doc.page_content
            if doc_key not in rrf_scores:
                rrf_scores[doc_key] = {"doc": doc, "score": 0.0}
            rrf_scores[doc_key]["score"] += 1.0 / (k + rank)

        # 按 RRF 分数降序排列
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

        return [item["doc"] for item in sorted_results]

    def search(self, query: str, top_k: int = None) -> list[Document]:
        """
        混合检索入口：BM25 + Embedding → RRF 融合

        参数:
            query: 用户查询
            top_k: 返回结果数量

        返回:
            融合排序后的 Document 列表
        """
        if top_k is None:
            top_k = self.k

        # 自动检测 BM25 索引是否过期（ChromaDB 文档数变化时自动刷新）
        try:
            current_count = self.vector_store.get()["__metadata__"].get("count", None)
            if current_count is None:
                results = self.vector_store.get(include=["documents"])
                current_count = len(results["documents"]) if results and results["documents"] else 0
            if current_count != self._bm25_doc_count:
                logger.info(f"[BM25] 文档数变化 ({self._bm25_doc_count} → {current_count})，自动刷新索引")
                self._build_bm25_index()
        except Exception:
            pass

        # 两个通道各自多检索一些，融合后截断
        retrieve_k = top_k * 3

        bm25_results = self._bm25_search(query, retrieve_k)
        embedding_results = self._embedding_search(query, retrieve_k)

        logger.info(
            f"[混合检索] query=\"{query}\" | "
            f"BM25返回{len(bm25_results)}条 | Embedding返回{len(embedding_results)}条"
        )

        # RRF 融合
        fused_results = self._rrf_fusion(bm25_results, embedding_results)

        return fused_results[:top_k]

    def refresh_index(self):
        """刷新 BM25 索引（知识库更新后调用）"""
        self._build_bm25_index()
