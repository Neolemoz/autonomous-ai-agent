import argparse

from agent.queue import add_task


def main():
    parser = argparse.ArgumentParser(description="Add task to agent queue")
    parser.add_argument(
        "--mode",
        choices=["general", "coder", "tester", "planner"],
        default="general",
        help="Task execution mode",
    )
    parser.add_argument("task", type=str, help="Task description")

    args = parser.parse_args()

    task_payload = {
        "input": f"[MODE: {args.mode}]\n{args.task}",
        "mode": args.mode,
        "status": "pending",
        "retry_count": 0,
        "max_retry": 3,
        "last_error": "",
    }

    add_task(task_payload)
    print(f"[AGENT] Task added: {args.task}")


if __name__ == "__main__":
    main()
