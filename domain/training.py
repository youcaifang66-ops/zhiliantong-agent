from datetime import date, datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Goal(StrEnum):
    MUSCLE_GAIN = "增肌"
    FAT_LOSS = "减脂"
    FITNESS = "提升体能"
    MOBILITY = "改善活动度"


class ExerciseSet(BaseModel):
    exercise: str = Field(min_length=1, max_length=80)
    weight_kg: float | None = Field(default=None, ge=0, le=1000)
    reps: int | None = Field(default=None, ge=1, le=500)
    duration_seconds: int | None = Field(default=None, ge=1, le=86400)
    rpe: float = Field(ge=1, le=10)

    @model_validator(mode="after")
    def require_reps_or_duration(self):
        if self.reps is None and self.duration_seconds is None:
            raise ValueError("次数和持续时间至少填写一项")
        return self


class TrainingRecordCreate(BaseModel):
    request_id: str = Field(min_length=8, max_length=80)
    user_id: str = Field(min_length=1, max_length=64)
    training_date: date
    goal: Goal
    sets: list[ExerciseSet] = Field(min_length=1, max_length=100)
    notes: str = Field(default="", max_length=500)


class TrainingRecord(TrainingRecordCreate):
    id: int
    created_at: datetime


class TrainingSummary(BaseModel):
    user_id: str
    month: str
    sessions: int
    total_sets: int
    average_rpe: float
    total_volume_kg: float


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
