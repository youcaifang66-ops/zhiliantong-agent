import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TaskCheckpoint:
    task_id: str
    step: str
    status: str
    payload: dict
    attempts: int = 0


class SQLiteTaskStore:
    """长任务检查点；同一 task_id 更新时保持最近状态。"""

    def __init__(self, database: str | Path = ":memory:"):
        self.connection = sqlite3.connect(str(database), check_same_thread=False)
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS task_checkpoint (
            task_id TEXT PRIMARY KEY, step TEXT NOT NULL, status TEXT NOT NULL,
            payload TEXT NOT NULL, attempts INTEGER NOT NULL)"""
        )

    def save(self, checkpoint: TaskCheckpoint) -> None:
        values = asdict(checkpoint)
        values["payload"] = json.dumps(values["payload"], ensure_ascii=False)
        self.connection.execute(
            """INSERT INTO task_checkpoint VALUES (:task_id,:step,:status,:payload,:attempts)
            ON CONFLICT(task_id) DO UPDATE SET step=excluded.step,status=excluded.status,
            payload=excluded.payload,attempts=excluded.attempts""",
            values,
        )
        self.connection.commit()

    def load(self, task_id: str) -> TaskCheckpoint | None:
        row = self.connection.execute(
            "SELECT task_id,step,status,payload,attempts FROM task_checkpoint WHERE task_id=?",
            (task_id,),
        ).fetchone()
        return TaskCheckpoint(row[0], row[1], row[2], json.loads(row[3]), row[4]) if row else None

    def run_with_retry(self, checkpoint: TaskCheckpoint, operation, max_attempts: int = 3):
        last_error = None
        while checkpoint.attempts < max_attempts:
            checkpoint.attempts += 1
            checkpoint.status = "running"
            self.save(checkpoint)
            try:
                result = operation()
                checkpoint.status = "completed"
                checkpoint.payload["result"] = result
                self.save(checkpoint)
                return result
            except Exception as error:  # retry boundary intentionally catches tool failures
                last_error = error
                checkpoint.status = "retrying"
                checkpoint.payload["last_error"] = str(error)
                self.save(checkpoint)
        checkpoint.status = "failed"
        self.save(checkpoint)
        raise RuntimeError(f"任务 {checkpoint.task_id} 重试耗尽") from last_error
