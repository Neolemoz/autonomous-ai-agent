import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
QUEUE_PATH = BASE_DIR / "tasks" / "queue.json"


def _ensure_queue_file() -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not QUEUE_PATH.exists():
        QUEUE_PATH.write_text("[]", encoding="utf-8")


def load_queue() -> list[Any]:
    _ensure_queue_file()
    try:
        data = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = []
        QUEUE_PATH.write_text("[]", encoding="utf-8")

    if not isinstance(data, list):
        data = []
        QUEUE_PATH.write_text("[]", encoding="utf-8")

    return data


def save_queue(queue: list[Any]) -> None:
    _ensure_queue_file()
    QUEUE_PATH.write_text(json.dumps(queue, indent=2), encoding="utf-8")


def add_task(task: Any) -> None:
    queue = load_queue()
    queue.append(task)
    save_queue(queue)
