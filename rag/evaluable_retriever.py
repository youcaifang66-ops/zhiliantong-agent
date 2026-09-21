import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SYNONYMS = {
    "膝盖不舒服": "膝盖 疼痛 关节 安全 停止",
    "加重量": "渐进 超负荷 重量 次数",
    "练几次": "频率 每周 训练",
    "动作标准": "动作 要点 姿势 控制",
}


def tokenize(text: str) -> list[str]:
    text = re.sub(r"\s+", "", text.lower())
    return [text[index : index + 2] for index in range(max(1, len(text) - 1))]


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source: str
    content: str


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    source: str
    content: str
    rrf_score: float
    channels: tuple[str, ...]


class EvaluableHybridRetriever:
    """可离线回归的 BM25 + 稠密字符向量 + RRF 检索器。"""

    def __init__(self, chunks: list[KnowledgeChunk]):
        self.chunks = chunks
        self.documents = [tokenize(chunk.content) for chunk in chunks]
        self.document_frequency = Counter(
            token for document in self.documents for token in set(document)
        )
        self.average_length = sum(map(len, self.documents)) / max(1, len(self.documents))

    @classmethod
    def from_directory(cls, path: str | Path):
        chunks = []
        for file in sorted(Path(path).glob("*.txt")):
            for index, paragraph in enumerate(file.read_text(encoding="utf-8").split("\n\n")):
                content = paragraph.strip()
                if content:
                    chunks.append(KnowledgeChunk(f"{file.stem}-{index}", file.name, content))
        return cls(chunks)

    def _bm25(self, query: str):
        query_tokens = tokenize(query)
        scores = []
        for index, document in enumerate(self.documents):
            frequencies = Counter(document)
            score = 0.0
            for token in query_tokens:
                df = self.document_frequency[token]
                idf = math.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))
                tf = frequencies[token]
                denominator = tf + 1.5 * (1 - 0.75 + 0.75 * len(document) / self.average_length)
                score += idf * (tf * 2.5 / denominator if denominator else 0)
            scores.append((score, index))
        return [index for score, index in sorted(scores, reverse=True) if score > 0]

    @staticmethod
    def _cosine(left: Counter, right: Counter) -> float:
        numerator = sum(value * right[token] for token, value in left.items())
        denominator = math.sqrt(sum(v * v for v in left.values()) * sum(v * v for v in right.values()))
        return numerator / denominator if denominator else 0.0

    def _semantic(self, query: str):
        expanded = query + " " + " ".join(value for key, value in SYNONYMS.items() if key in query)
        query_vector = Counter(tokenize(expanded))
        scores = [
            (self._cosine(query_vector, Counter(document)), index)
            for index, document in enumerate(self.documents)
        ]
        return [index for score, index in sorted(scores, reverse=True) if score > 0]

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        ranks = {"bm25": self._bm25(query), "semantic": self._semantic(query)}
        fused: dict[int, dict] = {}
        for channel, indices in ranks.items():
            for rank, index in enumerate(indices[: top_k * 3], start=1):
                item = fused.setdefault(index, {"score": 0.0, "channels": []})
                item["score"] += 1 / (60 + rank)
                item["channels"].append(channel)
        results = []
        for index, item in sorted(fused.items(), key=lambda pair: pair[1]["score"], reverse=True)[:top_k]:
            chunk = self.chunks[index]
            results.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    content=chunk.content,
                    rrf_score=round(item["score"], 6),
                    channels=tuple(item["channels"]),
                )
            )
        return results
