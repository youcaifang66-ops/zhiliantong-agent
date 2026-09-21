import json
import logging
import time
import uuid
from contextlib import contextmanager


logger = logging.getLogger("zhiliantong.telemetry")


@contextmanager
def trace_step(step: str, trace_id: str | None = None):
    trace_id = trace_id or str(uuid.uuid4())
    started = time.perf_counter()
    try:
        yield trace_id
        status, error = "success", None
    except Exception as exc:
        status, error = "error", type(exc).__name__
        raise
    finally:
        logger.info(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "step": step,
                    "status": status,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error_type": error,
                },
                ensure_ascii=False,
            )
        )
