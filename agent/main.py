import os
import re
import sys
from datetime import datetime
from pathlib import Path

from agent.agent_loop import AutonomousAgent
from agent.evaluator import evaluate
from agent.planner import generate_plan, parse_plan, replan
from agent.tools.run_tests import run_tests


def _step_failed(result: str) -> bool:
    normalized = result.strip()
    return normalized.startswith("Stopped after") or normalized.startswith("Task failed")


def _is_test_execution_step(step: str) -> bool:
    lowered = step.lower()
    mentions_tests = "pytest" in lowered or "unittest" in lowered or "test_" in lowered
    runs_something = "run " in lowered or "execute " in lowered
    return mentions_tests and runs_something


def _extract_target_file(*texts: str) -> str | None:
    for text in texts:
        matches = re.findall(r"\b([A-Za-z0-9_\-]+\.py)\b", text)
        for match in matches:
            if match.startswith("test_") or match.endswith("_test.py"):
                continue
            return match
    return None


def find_test_command(target_file: str | None) -> str | None:
    if not target_file:
        return None

    stem = Path(target_file).stem
    candidates = [
        Path.cwd() / "tests" / f"test_{stem}.py",
        Path.cwd() / f"test_{stem}.py",
        Path.cwd() / "tests" / f"{stem}_test.py",
        Path.cwd() / f"{stem}_test.py",
    ]

    for test_file in candidates:
        if test_file.exists():
            try:
                relative_path = test_file.relative_to(Path.cwd())
            except ValueError:
                relative_path = test_file
            return f"python3 {relative_path}"

    return None


def run_task(task: str, max_attempts: int = 3) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
        return "OPENAI_API_KEY is not set."

    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_lines = [f"TASK:\n{task}", f"MAX ATTEMPTS: {max_attempts}"]

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    result = ""
    old_plan = ""
    previous_scratchpad = ""
    last_plan = ""
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        print(f"\n=== ATTEMPT {attempt} ===")
        log_lines.append(f"\n=== ATTEMPT {attempt} ===")

        if attempt == 1:
            plan_text = generate_plan(task)
        else:
            plan_text = replan(task, old_plan, previous_scratchpad)
        steps = parse_plan(plan_text)

        if not steps:
            steps = [task]
            plan_text = f"PLAN:\n1. {task}"

        print("=== PLAN ===")
        print(plan_text)
        log_lines.append("=== PLAN ===")
        log_lines.append(plan_text)
        last_plan = plan_text

        scratchpad = []
        agent = AutonomousAgent(model=model)

        for index, step in enumerate(steps, start=1):
            print(f"\n=== STEP {index} ===")
            print(step)
            log_lines.append(f"\n=== STEP {index} ===")
            log_lines.append(step)

            if _is_test_execution_step(step):
                print("=== STEP RESULT ===")
                print("Skipped test execution step. Tests are run only by system logic.")
                log_lines.append("=== STEP RESULT ===")
                log_lines.append("Skipped test execution step. Tests are run only by system logic.")
                scratchpad.append(
                    f"Step {index}: {step}\nResult: Skipped test execution step. Tests are run only by system logic."
                )
                continue

            context = "\n".join(scratchpad) if scratchpad else "No completed steps yet."
            step_with_context = (
                f"Overall task:\n{task}\n\n"
                f"Progress so far:\n{context}\n\n"
                f"Current step:\n{step}"
            )

            result = agent.run(step_with_context)
            scratchpad.append(f"Step {index}: {step}\nResult: {result}")

            print("=== STEP RESULT ===")
            print(result)
            log_lines.append("=== STEP RESULT ===")
            log_lines.append(result)
            last_error = result

            if _step_failed(result):
                break

        full_scratchpad = "\n\n".join(scratchpad)
        target_file = _extract_target_file(plan_text, full_scratchpad)
        test_command = find_test_command(target_file)
        if test_command:
            print("=== TEST COMMAND ===")
            print(test_command)
            log_lines.append("=== TEST COMMAND ===")
            log_lines.append(test_command)
            test_result = run_tests(test_command)
            scratchpad.append(
                f"Test command: {test_command}\n"
                f"Test stdout:\n{test_result['stdout']}\n"
                f"Test stderr:\n{test_result['stderr']}"
            )
            full_scratchpad = "\n\n".join(scratchpad)
            print("=== TEST RESULT ===")
            print(test_result["stdout"] or test_result["stderr"] or "(no output)")
            log_lines.append("=== TEST RESULT ===")
            log_lines.append(test_result["stdout"] or test_result["stderr"] or "(no output)")
            evaluation = "YES" if test_result["success"] else "NO"
            last_error = test_result["stderr"] or test_result["stdout"] or result
        else:
            evaluation = evaluate(task, full_scratchpad)

        print("=== EVALUATION ===")
        print(evaluation)
        print("\n=== FINAL ANSWER ===")
        print(result)
        log_lines.append("=== EVALUATION ===")
        log_lines.append(evaluation)
        log_lines.append("=== FINAL ANSWER ===")
        log_lines.append(result)

        if evaluation == "YES":
            print("FINAL: SUCCESS")
            log_lines.append("FINAL: SUCCESS")
            log_path.write_text("\n".join(log_lines))
            return result

        old_plan = plan_text
        previous_scratchpad = full_scratchpad

        if attempt < max_attempts:
            print("Evaluation failed. Retrying...")
            log_lines.append("Evaluation failed. Retrying...")
            continue

        print("FINAL: FAILED AFTER MAX ATTEMPTS")
        print("=== ERROR REPORT ===")
        print(f"Task: {task}")
        print(f"Last plan:\n{last_plan}")
        print(f"Last error:\n{last_error}")
        print("Suggestion: Try refining the task or increasing max_attempts")
        log_lines.append("FINAL: FAILED AFTER MAX ATTEMPTS")
        log_lines.append("=== ERROR REPORT ===")
        log_lines.append(f"Task: {task}")
        log_lines.append(f"Last plan:\n{last_plan}")
        log_lines.append(f"Last error:\n{last_error}")
        log_lines.append("Suggestion: Try refining the task or increasing max_attempts")
        log_path.write_text("\n".join(log_lines))
        return result

    log_path.write_text("\n".join(log_lines))
    return result


def main():
    task = " ".join(sys.argv[1:]).strip()
    if not task:
        task = input("Enter task: ").strip()

    if not task:
        print("No task provided.")
        return

    run_task(task)


if __name__ == "__main__":
    main()
