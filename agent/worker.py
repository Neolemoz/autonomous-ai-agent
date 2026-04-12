import argparse
import io
import json
import time
import traceback
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.logger import log
from agent.main import run_task


TASKS_DIR = Path("tasks")
QUEUE_PATH = TASKS_DIR / "queue.json"
HISTORY_PATH = TASKS_DIR / "history.json"
POLL_INTERVAL_SECONDS = 1.5


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds") + "Z"


def _ensure_json_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]\n", encoding="utf-8")


def _ensure_queue_file() -> None:
    _ensure_json_file(QUEUE_PATH)


def _ensure_history_file() -> None:
    _ensure_json_file(HISTORY_PATH)


def _load_queue() -> list[dict[str, Any]]:
    _ensure_queue_file()
    try:
        data = json.loads(QUEUE_PATH.read_text(encoding="utf-8") or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid queue JSON: {error}") from error

    if not isinstance(data, list):
        raise ValueError("Queue file must contain a JSON list.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            normalized.append(
                {
                    "id": index + 1,
                    "input": item,
                    "status": "pending",
                    "retry_count": 0,
                    "max_retry": 3,
                    "last_error": "",
                }
            )
            continue

        if not isinstance(item, dict):
            raise ValueError(f"Queue item at index {index} must be an object or string.")

        task = dict(item)
        task.setdefault("id", index + 1)
        if "input" not in task:
            raise ValueError(f"Queue item at index {index} is missing 'input'.")
        task["input"] = str(task["input"])
        task["status"] = str(task.get("status", "pending"))
        task["retry_count"] = int(task.get("retry_count", 0))
        task["max_retry"] = int(task.get("max_retry", 3))
        task["last_error"] = str(task.get("last_error", ""))
        normalized.append(task)

    return normalized


def _save_queue(queue: list[dict[str, Any]]) -> None:
    _ensure_queue_file()
    QUEUE_PATH.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")


def _append_history(task: str, status: str, result: str) -> None:
    _ensure_history_file()
    try:
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8") or "[]")
    except json.JSONDecodeError:
        history = []

    if not isinstance(history, list):
        history = []

    history.append(
        {
            "task": task,
            "status": status,
            "result": result,
            "timestamp": _utc_now(),
        }
    )
    HISTORY_PATH.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def _find_next_pending_task(queue: list[dict[str, Any]]) -> dict[str, Any] | None:
    for task in queue:
        if task.get("status") == "pending":
            return task
    return None


def _latest_run_log(before: set[Path]) -> Path | None:
    logs_dir = Path("logs")
    candidates = sorted(logs_dir.glob("run_*.log"), key=lambda path: path.stat().st_mtime)
    for path in reversed(candidates):
        if path not in before:
            return path
    return candidates[-1] if candidates else None


def _execution_succeeded(result: str, log_path: Path | None) -> bool:
    normalized = result.strip()
    if normalized.startswith("OPENAI_API_KEY is not set."):
        return False
    if normalized.startswith("Stopped after") or normalized.startswith("Task failed"):
        return False

    if log_path and log_path.exists():
        log_text = log_path.read_text(encoding="utf-8")
        if "FINAL: SUCCESS" in log_text:
            return True
        if "FINAL: FAILED AFTER MAX ATTEMPTS" in log_text:
            return False

    return bool(normalized)


def _run_task_with_status(task_input: str, max_retry: int) -> tuple[bool, str]:
    logs_dir = Path("logs")
    before_logs = set(logs_dir.glob("run_*.log")) if logs_dir.exists() else set()
    captured_stdout = io.StringIO()
    with redirect_stdout(captured_stdout):
        result = run_task(task_input, max_attempts=max_retry)
    _replay_agent_output(captured_stdout.getvalue())
    log_path = _latest_run_log(before_logs)
    success = _execution_succeeded(result, log_path)

    if success:
        return True, result.strip()

    if log_path and log_path.exists():
        log_text = log_path.read_text(encoding="utf-8")
        marker = "Last error:\n"
        if marker in log_text:
            return False, log_text.split(marker, 1)[1].strip()
    return False, result.strip() or "Task execution failed."


def _replay_agent_output(output: str) -> None:
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("[AGENT] "):
            log(line)


def _process_one_task(queue: list[dict[str, Any]], task: dict[str, Any]) -> None:
    task["status"] = "running"
    task["started_at"] = _utc_now()
    task["last_error"] = ""
    _save_queue(queue)
    log("[AGENT] Task started")
    log("[AGENT] Planning...")

    try:
        success, result_message = _run_task_with_status(
            task_input=task["input"],
            max_retry=max(1, int(task.get("max_retry", 3))),
        )
        if success:
            task["status"] = "completed"
            task["completed_at"] = _utc_now()
            task["last_error"] = ""
            _save_queue(queue)
            _append_history(task["input"], "completed", result_message)
            log("[AGENT] SUCCESS")
            return

        task["retry_count"] = int(task.get("retry_count", 0)) + 1
        task["last_error"] = result_message
        if task["retry_count"] >= int(task.get("max_retry", 3)):
            task["status"] = "failed"
            task["failed_at"] = _utc_now()
            _save_queue(queue)
            _append_history(task["input"], "failed", result_message)
            log(f"[AGENT] FAILED: {result_message}")
        else:
            task["status"] = "pending"
            _save_queue(queue)
            log(f"[AGENT] FAILED: {result_message}")
    except Exception as error:
        task["retry_count"] = int(task.get("retry_count", 0)) + 1
        task["last_error"] = f"{error}\n{traceback.format_exc()}".strip()
        if task["retry_count"] >= int(task.get("max_retry", 3)):
            task["status"] = "failed"
            task["failed_at"] = _utc_now()
            _save_queue(queue)
            _append_history(task["input"], "failed", task["last_error"])
        else:
            task["status"] = "pending"
            _save_queue(queue)
        log(f"[AGENT] FAILED: {error}")


def run_worker(*, once: bool = False) -> None:
    log("[AGENT] Worker started")
    if once:
        log("[AGENT] Running in ONCE mode")
        try:
            queue = _load_queue()
            task = _find_next_pending_task(queue)
            if task is not None:
                _process_one_task(queue, task)
        except Exception as error:
            log(f"[AGENT] FAILED: {error}")
            try:
                _ensure_queue_file()
                _ensure_history_file()
            except Exception:
                pass
        log("[AGENT] Done (once mode)")
        return

    while True:
        try:
            queue = _load_queue()
            task = _find_next_pending_task(queue)
            if task is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            _process_one_task(queue, task)
        except Exception as error:
            log(f"[AGENT] FAILED: {error}")
            try:
                _ensure_queue_file()
                _ensure_history_file()
            except Exception:
                pass
            time.sleep(POLL_INTERVAL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_worker(once=args.once)


if __name__ == "__main__":
    main()
