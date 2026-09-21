import json
import sqlite3
from pathlib import Path

from domain.training import TrainingRecord, TrainingRecordCreate, TrainingSummary, utc_now


class SQLiteTrainingRepository:
    """带请求幂等键的训练记录仓储。"""

    def __init__(self, database: str | Path = ":memory:"):
        self.connection = sqlite3.connect(str(database), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS training_record (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              request_id TEXT NOT NULL UNIQUE,
              user_id TEXT NOT NULL,
              training_date TEXT NOT NULL,
              goal TEXT NOT NULL,
              sets_json TEXT NOT NULL,
              notes TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )

    def save(self, command: TrainingRecordCreate) -> tuple[TrainingRecord, bool]:
        existing = self.connection.execute(
            "SELECT * FROM training_record WHERE request_id = ?", (command.request_id,)
        ).fetchone()
        if existing:
            return self._to_record(existing), False

        created_at = utc_now()
        cursor = self.connection.execute(
            """INSERT INTO training_record
            (request_id,user_id,training_date,goal,sets_json,notes,created_at)
            VALUES (?,?,?,?,?,?,?)""",
            (
                command.request_id,
                command.user_id,
                command.training_date.isoformat(),
                command.goal.value,
                json.dumps([item.model_dump() for item in command.sets], ensure_ascii=False),
                command.notes,
                created_at.isoformat(),
            ),
        )
        self.connection.commit()
        return TrainingRecord(id=cursor.lastrowid, created_at=created_at, **command.model_dump()), True

    def monthly_summary(self, user_id: str, month: str) -> TrainingSummary:
        rows = self.connection.execute(
            "SELECT * FROM training_record WHERE user_id=? AND substr(training_date,1,7)=?",
            (user_id, month),
        ).fetchall()
        sets = [item for row in rows for item in json.loads(row["sets_json"])]
        average_rpe = round(sum(item["rpe"] for item in sets) / len(sets), 2) if sets else 0.0
        volume = sum((item.get("weight_kg") or 0) * (item.get("reps") or 0) for item in sets)
        return TrainingSummary(
            user_id=user_id,
            month=month,
            sessions=len(rows),
            total_sets=len(sets),
            average_rpe=average_rpe,
            total_volume_kg=round(volume, 2),
        )

    @staticmethod
    def _to_record(row: sqlite3.Row) -> TrainingRecord:
        return TrainingRecord.model_validate(
            {
                **dict(row),
                "sets": json.loads(row["sets_json"]),
            }
        )
