from infrastructure.task_store import SQLiteTaskStore, TaskCheckpoint


def test_checkpoint_retry_and_resume():
    store = SQLiteTaskStore()
    checkpoint = TaskCheckpoint("task-1", "save_record", "pending", {"request_id": "request-1"})
    calls = 0

    def flaky_operation():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TimeoutError("temporary timeout")
        return {"saved": True}

    assert store.run_with_retry(checkpoint, flaky_operation) == {"saved": True}
    restored = store.load("task-1")
    assert restored.status == "completed"
    assert restored.attempts == 3
    assert restored.payload["result"] == {"saved": True}
