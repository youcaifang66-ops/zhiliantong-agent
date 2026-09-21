from datetime import date

from domain.training import ExerciseSet, Goal, TrainingRecordCreate
from infrastructure.training_repository import SQLiteTrainingRepository


def command(request_id="request-001"):
    return TrainingRecordCreate(
        request_id=request_id,
        user_id="1001",
        training_date=date(2026, 9, 21),
        goal=Goal.MUSCLE_GAIN,
        sets=[
            ExerciseSet(exercise="深蹲", weight_kg=60, reps=8, rpe=7),
            ExerciseSet(exercise="深蹲", weight_kg=60, reps=8, rpe=8),
        ],
    )


def test_idempotent_save_and_summary():
    repository = SQLiteTrainingRepository()
    first, created = repository.save(command())
    second, duplicated = repository.save(command())
    assert created is True and duplicated is False
    assert first.id == second.id
    assert repository.monthly_summary("1001", "2026-09").model_dump() == {
        "user_id": "1001",
        "month": "2026-09",
        "sessions": 1,
        "total_sets": 2,
        "average_rpe": 7.5,
        "total_volume_kg": 960.0,
    }
