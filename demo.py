from agent.queue import add_task
from agent.worker import _find_next_pending_task, _load_queue, _process_one_task


def main() -> None:
    task = "create a python script that prints hello"

    print("[DEMO] Starting agent demo")
    add_task(task)
    print("[DEMO] Task added")

    queue = _load_queue()
    pending_task = _find_next_pending_task(queue)

    if pending_task is None:
        print("[DEMO] No pending task found")
        return

    _process_one_task(queue, pending_task)
    print("[DEMO] Done")


if __name__ == "__main__":
    main()
