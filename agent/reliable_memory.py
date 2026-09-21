import hashlib
import math
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _tokens(text: str) -> set[str]:
    normalized = "".join(text.lower().split())
    return {normalized[index : index + 2] for index in range(max(1, len(normalized) - 1))}


def _similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a | b else 0.0


class ReliableMemory:
    """短期窗口 + 长期事实 + 会话摘要的三层记忆。"""

    def __init__(self, database: str | Path = ":memory:", window_size: int = 10):
        self.window_size = window_size
        self.windows = defaultdict(lambda: deque(maxlen=window_size * 2))
        self.connection = sqlite3.connect(str(database), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_fact (
              user_id TEXT NOT NULL, fact_hash TEXT NOT NULL, content TEXT NOT NULL,
              created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
              PRIMARY KEY(user_id, fact_hash)
            );
            CREATE TABLE IF NOT EXISTS memory_summary (
              user_id TEXT PRIMARY KEY, content TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            """
        )

    def append_message(self, user_id: str, role: str, content: str) -> None:
        self.windows[user_id].append({"role": role, "content": content})

    def short_term(self, user_id: str) -> list[dict[str, str]]:
        return list(self.windows[user_id])

    def remember_fact(self, user_id: str, content: str, ttl_days: int = 180) -> bool:
        normalized = " ".join(content.split())
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        now = datetime.now(timezone.utc)

        existing = self.connection.execute(
            "SELECT content FROM memory_fact WHERE user_id=?", (user_id,)
        ).fetchall()
        if any(_similarity(normalized, row["content"]) >= 0.88 for row in existing):
            return False
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO memory_fact VALUES (?,?,?,?,?)",
            (user_id, digest, normalized, now.isoformat(), (now + timedelta(days=ttl_days)).isoformat()),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def recall(self, user_id: str, query: str, limit: int = 3, now: datetime | None = None):
        now = now or datetime.now(timezone.utc)
        rows = self.connection.execute(
            "SELECT content,created_at,expires_at FROM memory_fact WHERE user_id=?", (user_id,)
        ).fetchall()
        ranked = []
        for row in rows:
            if datetime.fromisoformat(row["expires_at"]) <= now:
                continue
            age_days = max(0.0, (now - datetime.fromisoformat(row["created_at"])).total_seconds() / 86400)
            semantic = _similarity(query, row["content"])
            decay = math.exp(-age_days / 90)
            ranked.append((semantic * 0.8 + decay * 0.2, row["content"]))
        return [content for _, content in sorted(ranked, reverse=True)[:limit]]

    def save_summary(self, user_id: str, content: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            """INSERT INTO memory_summary VALUES (?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET content=excluded.content,updated_at=excluded.updated_at""",
            (user_id, content, now),
        )
        self.connection.commit()

    def summary(self, user_id: str) -> str:
        row = self.connection.execute(
            "SELECT content FROM memory_summary WHERE user_id=?", (user_id,)
        ).fetchone()
        return row["content"] if row else ""
