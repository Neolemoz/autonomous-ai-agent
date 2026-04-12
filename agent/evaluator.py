import os
import re

from openai import OpenAI


def _normalize_output(value: str) -> str:
    return value.strip()


def _looks_like_real_error(stderr: str) -> bool:
    normalized = stderr.strip()
    if not normalized:
        return False

    error_markers = (
        "traceback",
        "error",
        "exception",
        "failed",
        "no such file",
        "syntaxerror",
        "nameerror",
        "typeerror",
        "valueerror",
        "modulenotfounderror",
    )
    lowered = normalized.lower()
    return any(marker in lowered for marker in error_markers)


def _log_debug(exit_code: int | None, stdout: str, stderr: str) -> None:
    print("EVALUATION DEBUG:")
    print(f"exit_code={exit_code}")
    print(f"stdout={stdout!r}")
    print(f"stderr={stderr!r}")


def _log_result(success: bool, reason: str = "") -> str:
    if success:
        print("EVALUATION: SUCCESS")
        return "YES"
    print(f"EVALUATION: FAILED - reason: {reason}")
    return "NO"


def _extract_last_command_result(scratchpad: str) -> tuple[int | None, str, str]:
    matches = re.findall(
        r"exit_code=(\d+)\nstdout:\n(.*?)(?=\nstderr:\n)(?:\nstderr:\n)(.*?)(?=\n\n|$)",
        scratchpad,
        re.DOTALL,
    )
    if not matches:
        return None, "", ""

    exit_code, stdout, stderr = matches[-1]
    return int(exit_code), stdout, stderr


def _llm_fallback(task: str, scratchpad: str) -> str:
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Task:\n{task}\n\n"
                    f"Execution log:\n{scratchpad}\n\n"
                    "Question:\n"
                    "Was the task fully completed correctly?\n\n"
                    "Rules:\n"
                    "* Output ONLY: YES or NO\n"
                    "* Prefer NO if the task may be incomplete"
                ),
            }
        ],
        temperature=0,
    )
    result = (response.choices[0].message.content or "").strip().upper()
    if result == "YES":
        print("EVALUATION: SUCCESS")
        return "YES"
    print("EVALUATION: FAILED - reason: fallback evaluation returned NO")
    return "NO"


def evaluate(task: str, scratchpad: str) -> str:
    exit_code, stdout, stderr = _extract_last_command_result(scratchpad)
    normalized_stdout = _normalize_output(stdout)
    _log_debug(exit_code, stdout, stderr)

    if exit_code is not None:
        if exit_code != 0:
            return _log_result(False, f"exit_code={exit_code}")

        if _looks_like_real_error(stderr):
            return _log_result(False, "stderr contains critical error output")

        if normalized_stdout == "":
            return _log_result(False, "stdout is empty")

        lowered_task = task.lower()
        if "print" in lowered_task or "output" in lowered_task:
            if normalized_stdout == "":
                return _log_result(False, "expected visible output but stdout is empty")

        return _log_result(True)

    return _llm_fallback(task, scratchpad)
