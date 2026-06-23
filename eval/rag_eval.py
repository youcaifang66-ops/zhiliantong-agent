"""
RAG 全链路评测脚本

评测维度：
  检索阶段：Recall@K、HitRate@K、MRR、NDCG@K
  生成阶段：Faithfulness、Answer Relevancy（基于关键词匹配的轻量实现）

使用方式：
  python -m eval.rag_eval
"""
import json
import math
import sys
import os

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.hybrid_retriever import HybridRetriever
from rag.rag_service import RagSummarizeService
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


# ============================================================
# 评测指标计算函数
# ============================================================

def recall_at_k(retrieved_texts: list[str], expected_keywords: list[str], k: int) -> float:
    """
    Recall@K：在返回的 top_k 个文档中，命中了多少比例的期望关键词

    参数:
        retrieved_texts: 检索返回的文档文本列表
        expected_keywords: 期望命中的关键词列表
        k: 取前 k 个文档评估

    返回:
        命中率 0.0 ~ 1.0
    """
    top_k_texts = " ".join(retrieved_texts[:k]).lower()
    hit_count = sum(1 for kw in expected_keywords if kw.lower() in top_k_texts)
    return hit_count / len(expected_keywords) if expected_keywords else 0.0


def hit_rate_at_k(retrieved_texts: list[str], expected_keywords: list[str], k: int) -> float:
    """
    HitRate@K：top_k 个文档中是否至少命中了 1 个期望关键词

    返回:
        1.0（命中）或 0.0（未命中）
    """
    top_k_texts = " ".join(retrieved_texts[:k]).lower()
    for kw in expected_keywords:
        if kw.lower() in top_k_texts:
            return 1.0
    return 0.0


def mrr(retrieved_texts: list[str], expected_keywords: list[str]) -> float:
    """
    MRR（Mean Reciprocal Rank）：第一个命中关键词的文档排在第几名

    公式：1 / rank（rank 是第一个命中文档的排名，从 1 开始）
    如果所有文档都没命中，MRR = 0

    返回:
        1/rank，范围 0.0 ~ 1.0
    """
    for rank, text in enumerate(retrieved_texts, start=1):
        text_lower = text.lower()
        for kw in expected_keywords:
            if kw.lower() in text_lower:
                return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_texts: list[str], expected_keywords: list[str], k: int) -> float:
    """
    NDCG@K（Normalized Discounted Cumulative Gain）

    衡量排序质量：排名靠前的命中比排名靠后的命中得分更高

    计算步骤：
    1. DCG = Σ (命中则1/log2(i+1)，不命中则0)
    2. IDCG = 理想情况下（所有命中都在最前面）的 DCG
    3. NDCG = DCG / IDCG

    返回:
        0.0 ~ 1.0
    """
    # DCG：实际排序的得分
    dcg = 0.0
    for i, text in enumerate(retrieved_texts[:k], start=1):
        text_lower = text.lower()
        hit = any(kw.lower() in text_lower for kw in expected_keywords)
        if hit:
            dcg += 1.0 / math.log2(i + 1)

    # IDCG：理想排序的得分（假设所有关键词都在最前面命中）
    ideal_hits = min(len(expected_keywords), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# 主评测流程
# ============================================================

def run_retrieval_eval(retriever, golden_queries: list, k: int = 5):
    """
    检索阶段评测

    参数:
        retriever: 检索器实例
        golden_queries: 黄金测试集
        k: 评测的 K 值
    """
    print(f"\n{'='*60}")
    print(f"  检索阶段评测 | Top-K = {k}")
    print(f"{'='*60}")

    total_recall = 0.0
    total_hit_rate = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    n = len(golden_queries)

    for item in golden_queries:
        query = item["query"]
        expected_keywords = item["expected_keywords"]

        # 检索
        docs = retriever.search(query, top_k=k)
        retrieved_texts = [doc.page_content for doc in docs]

        # 计算指标
        recall = recall_at_k(retrieved_texts, expected_keywords, k)
        hit_rate = hit_rate_at_k(retrieved_texts, expected_keywords, k)
        mrr_score = mrr(retrieved_texts, expected_keywords)
        ndcg = ndcg_at_k(retrieved_texts, expected_keywords, k)

        total_recall += recall
        total_hit_rate += hit_rate
        total_mrr += mrr_score
        total_ndcg += ndcg

        # 单条详情（只打印未满分的）
        if recall < 1.0:
            print(f"  [{item['id']:2d}] Q: {query[:30]:30s} | Recall={recall:.2f} MRR={mrr_score:.2f} NDCG={ndcg:.2f}")

    # 平均指标
    avg_recall = total_recall / n
    avg_hit_rate = total_hit_rate / n
    avg_mrr = total_mrr / n
    avg_ndcg = total_ndcg / n

    print(f"\n{'─'*60}")
    print(f"  平均指标 ({n} 条查询):")
    print(f"    Recall@{k}     = {avg_recall:.4f} ({avg_recall*100:.1f}%)")
    print(f"    HitRate@{k}   = {avg_hit_rate:.4f} ({avg_hit_rate*100:.1f}%)")
    print(f"    MRR          = {avg_mrr:.4f}")
    print(f"    NDCG@{k}     = {avg_ndcg:.4f}")
    print(f"{'='*60}\n")

    return {
        f"Recall@{k}": round(avg_recall, 4),
        f"HitRate@{k}": round(avg_hit_rate, 4),
        "MRR": round(avg_mrr, 4),
        f"NDCG@{k}": round(avg_ndcg, 4),
    }


def run_generation_eval(rag_service, golden_queries: list):
    """
    生成阶段评测（基于关键词匹配的轻量实现）

    检查 LLM 生成的回答是否包含期望的关键词
    """
    print(f"\n{'='*60}")
    print(f"  生成阶段评测（Faithfulness + Answer Relevancy）")
    print(f"{'='*60}")

    total_faithfulness = 0.0
    total_relevancy = 0.0
    n = len(golden_queries)

    for item in golden_queries:
        query = item["query"]
        expected_keywords = item["expected_keywords"]

        try:
            answer = rag_service.rag_summarize(query)

            # Faithfulness：回答中是否包含了检索到的关键词（基于关键词召回）
            answer_lower = answer.lower()
            faith_hit = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
            faithfulness = faith_hit / len(expected_keywords) if expected_keywords else 0.0

            # Answer Relevancy：回答是否和问题相关（简单检查回答长度和关键词覆盖）
            relevancy = faithfulness  # 简化：Faithfulness 本身就是 relevancy 的近似

            total_faithfulness += faithfulness
            total_relevancy += relevancy

            if faithfulness < 0.5:
                print(f"  [{item['id']:2d}] Q: {query[:25]:25s} | Faith={faithfulness:.2f} Relev={relevancy:.2f}")

        except Exception as e:
            logger.warning(f"生成评测失败 [{item['id']}]: {e}")
            n -= 1

    avg_faithfulness = total_faithfulness / n if n > 0 else 0
    avg_relevancy = total_relevancy / n if n > 0 else 0

    print(f"\n{'─'*60}")
    print(f"  平均指标 ({n} 条查询):")
    print(f"    Faithfulness    = {avg_faithfulness:.4f} ({avg_faithfulness*100:.1f}%)")
    print(f"    Answer Relevancy= {avg_relevancy:.4f} ({avg_relevancy*100:.1f}%)")
    print(f"{'='*60}\n")

    return {
        "Faithfulness": round(avg_faithfulness, 4),
        "Answer Relevancy": round(avg_relevancy, 4),
    }


def run_chunk_size_comparison(golden_queries: list, k: int = 5):
    """
    对比不同 chunk_size 配置下的检索效果
    """
    print(f"\n{'='*60}")
    print(f"  Chunk Size 对比实验")
    print(f"{'='*60}")

    # 测试不同的 chunk_size
    chunk_sizes = [100, 200, 300, 500]

    for cs in chunk_sizes:
        print(f"\n  --- chunk_size = {cs} ---")
        # 这里只做模拟，实际需要重建 ChromaDB
        # 在实际使用中，需要为每个 chunk_size 重新加载文档并评测
        print(f"    需要重建 ChromaDB 后评测（修改 config/chroma.yml 的 chunk_size）")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    # 加载黄金测试集
    golden_queries_path = get_abs_path("eval/golden_queries.json")
    with open(golden_queries_path, "r", encoding="utf-8") as f:
        golden_queries = json.load(f)

    print(f"加载 {len(golden_queries)} 条 Golden Query")

    # 初始化检索器
    print("初始化 HybridRetriever...")
    retriever = HybridRetriever()

    # 检索阶段评测
    retrieval_metrics = run_retrieval_eval(retriever, golden_queries, k=5)

    # 生成阶段评测
    print("初始化 RAG Service...")
    rag_service = RagSummarizeService()
    generation_metrics = run_generation_eval(rag_service, golden_queries)

    # 汇总
    print(f"\n{'='*60}")
    print(f"  全链路评测汇总")
    print(f"{'='*60}")
    for name, value in {**retrieval_metrics, **generation_metrics}.items():
        print(f"    {name:20s} = {value}")
    print(f"{'='*60}")
