from agent.reliable_memory import ReliableMemory


def test_three_layer_memory_dedup_and_recall():
    memory = ReliableMemory(window_size=2)
    for index in range(6):
        memory.append_message("u1", "user", f"message-{index}")
    assert [item["content"] for item in memory.short_term("u1")] == [
        "message-2", "message-3", "message-4", "message-5"
    ]

    assert memory.remember_fact("u1", "用户深蹲工作组重量为60kg") is True
    assert memory.remember_fact("u1", "用户深蹲工作组重量为60kg") is False
    assert memory.recall("u1", "深蹲重量")[0] == "用户深蹲工作组重量为60kg"

    memory.save_summary("u1", "目标是增肌，每周训练三次")
    assert memory.summary("u1") == "目标是增肌，每周训练三次"
