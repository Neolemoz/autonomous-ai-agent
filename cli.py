import argparse
import json
from pathlib import Path

from main import run_task


def _load_config() -> dict:
    config_path = Path("config.json")
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return {}


def _print_history() -> None:
    history_path = Path(".history")
    if not history_path.exists():
        print("No history found.")
        return
    lines = [line.strip() for line in history_path.read_text().splitlines() if line.strip()]
    for item in lines[-10:]:
        print(item)


def _append_history(task: str) -> None:
    history_path = Path(".history")
    with history_path.open("a") as handle:
        handle.write(f"{task}\n")


config = _load_config()

parser = argparse.ArgumentParser(description="AI Agent CLI")
parser.add_argument("task", nargs="?", type=str, help="Task description")
parser.add_argument("--max_attempts", type=int, default=None)
parser.add_argument("--history", action="store_true")

args = parser.parse_args()

if args.history:
    _print_history()
else:
    if not args.task:
        parser.error("task is required unless --history is used")
    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = config.get("max_attempts", 3)
    run_task(args.task, max_attempts=max_attempts)
    _append_history(args.task)
