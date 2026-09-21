from utils.structured_log import trace_step


def test_trace_id_is_stable_inside_step():
    with trace_step("retrieval", "trace-1") as trace_id:
        assert trace_id == "trace-1"
