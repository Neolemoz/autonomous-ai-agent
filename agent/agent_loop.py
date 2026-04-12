import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.logger import log
from agent.memory import AgentMemory
from agent.planner import extract_task_requirements
from agent.safety import SafetyManager
from agent.tools.file import FileTool
from agent.tools.python_exec import PythonExecTool
from agent.tools.shell import ShellTool


MAX_STEPS = 15
MAX_RETRIES = 3


SYSTEM_PROMPT = """You are an autonomous coding agent.

You must solve the user's task by following a ReAct loop:
Thought: briefly explain the next step
Action: one of SHELL, FILE_READ, FILE_WRITE, FILE_APPEND, PYTHON, FINAL_ANSWER
Action Input: the input for that action

You MUST strictly follow the required format.
Do NOT output FINAL_ANSWER directly.
You MUST always output:
Thought, Action, Action Input.
Never output Thought alone.
If unsure, choose a reasonable action.
Do NOT return FINAL_ANSWER unless you have executed at least one valid action.
If the task is unclear, try a reasonable default action.
Never ask the user for more information.

Rules:
- Use exactly one action per response.
- When using FILE_WRITE or FILE_APPEND, Action Input must be JSON with keys "path" and "content".
- When using FILE_READ, Action Input must be just the file path.
- When using SHELL, Action Input must be a single shell command.
- When using PYTHON, Action Input must be Python code only.
- When the task is complete, return FINAL_ANSWER with a concise summary.
- Base each next step on the latest observation.
- Do not invent observations.
- Minimize work. Do not create tests, README files, docs, or extra files unless the user explicitly asked for them.
- If a tool execution fails, treat that as a problem to fix.
- Prefer correcting the command or file and rerunning it.
- Do not treat a failed run as a completed task.
- When an action fails, you MUST follow the correct recovery strategy:
  1. If the error is SyntaxError or code-related:
     - You MUST: read the file, fix the code, write the corrected file, and run again.
     - Do NOT retry the same run command without fixing the code.
  2. If the error is command not found:
     - Try a sensible alternative command such as python3 instead of python.
  3. If the error is file not found:
     - Do NOT retry the same read command. Report the failure clearly or inspect the directory once if useful.
  4. Never try random variations of the same failing command.
"""


class AutonomousAgent:
    def __init__(self, model: str = "gpt-4.1-mini", max_steps: int = MAX_STEPS):
        self.model = model
        self.max_steps = max_steps
        self.client = OpenAI()
        self.memory = AgentMemory()
        self.safety = SafetyManager()

        class SimpleLogger:
            def info(self, msg, *args):
                log(msg % args if args else msg)

            def exception(self, msg, *args):
                log(f"ERROR: {msg % args if args else msg}")

        logger = SimpleLogger()

        self.tools = {
            "SHELL": ShellTool(self.safety, logger),
            "FILE_READ": FileTool(logger),
            "FILE_WRITE": FileTool(logger),
            "FILE_APPEND": FileTool(logger),
            "PYTHON": PythonExecTool(logger, self.safety),
        }

    def run(self, user_task: str) -> str:
        self._agent_log("Task started")
        self.memory.add("User Task", user_task)

        overall_task = self._extract_overall_task(user_task)
        current_step = self._extract_current_step(user_task)
        controls = self._build_controls(overall_task, current_step)

        if self._step_is_unnecessary(current_step, controls):
            observation = f"Skipped unnecessary step: {current_step}"
            self._agent_log(observation)
            self.memory.add("Observation", observation)
            return observation

        self._agent_log(f"Executing step: {current_step}")

        scratchpad = [
            f"User Task:\n{user_task}",
            f"Goal:\n{controls['goal']}",
            f"Constraints:\n" + "\n".join(f"- {item}" for item in controls["constraints"]),
            f"Minimal Output:\n{controls['minimal_output']}",
        ]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User Task:\n{user_task}\n\n"
                    f"Goal:\n{controls['goal']}\n\n"
                    "Constraints:\n"
                    + "\n".join(f"- {item}" for item in controls["constraints"])
                    + "\n\nMinimal Output:\n"
                    + controls["minimal_output"]
                ),
            },
        ]
        fallback_message = (
            "You must always take an action.\n"
            "If unsure, try a reasonable default action.\n"
            "Use the minimum work needed.\n"
            "Never respond with only Thought."
        )
        retry_count = 0
        last_error = None
        last_failed_action = None
        recovery_mode = None
        recovery_file_path = None

        while retry_count < MAX_RETRIES:
            prompt = self._build_prompt(scratchpad)

            model_output = self._call_llm(messages, prompt)
            messages.append({"role": "assistant", "content": model_output})

            if model_output.strip().startswith("FINAL_ANSWER:"):
                retry_count += 1
                last_error = "FINAL_ANSWER is not allowed here. Execute exactly one tool action and stop."
                messages.append({"role": "user", "content": last_error})
                scratchpad.append(f"Error:\n{last_error}")
                continue

            try:
                parsed = self._parse_output(model_output)
            except Exception as error:
                retry_count += 1
                last_error = str(error)
                self._agent_log(f"FAILED: {last_error}")
                self.memory.add("Observation", f"ERROR: {last_error}")
                scratchpad.append(f"Error:\n{last_error}")

                if retry_count >= MAX_RETRIES:
                    return f"Stopped after max retries. Last error: {last_error}"

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response format was invalid.\n"
                            "You MUST follow this format exactly:\n\n"
                            "Thought: ...\n"
                            "Action: ...\n"
                            "Action Input: ...\n\n"
                            "Or FINAL format:\n"
                            "Action: FINAL_ANSWER\n"
                            "Action Input: ..."
                        ),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": fallback_message,
                    }
                )
                continue

            thought = parsed["thought"]
            action = parsed["action"]
            action_input = parsed["action_input"]

            self.memory.add("Thought", thought)
            self.memory.add("Action", action)
            self.memory.add("Action Input", action_input)

            self._log_action(action, action_input)

            normalized_action_input = action_input.strip().strip('"').strip("'")
            if recovery_mode == "syntax_read":
                if action != "FILE_READ" or (
                    recovery_file_path and normalized_action_input != recovery_file_path
                ):
                    retry_count += 1
                    last_error = (
                        "Syntax-error recovery requires reading the failing file before any rerun."
                    )
                    failure_feedback = (
                        "The previous action failed.\n\n"
                        "Failure details:\n\n"
                        f"* Action: {action}\n"
                        f"* Input: {action_input}\n"
                        f"* Error: {last_error}\n\n"
                        "Fix the root cause instead of retrying blindly.\n"
                        f"You must read {recovery_file_path or 'the failing file'} first, then fix it, then write it, then run again.\n"
                        f"Your next action must be:\nAction: FILE_READ\nAction Input: {recovery_file_path or 'the failing file'}\n"
                        "If you execute the same command again without fixing the root cause, it will be considered a failure."
                    )
                    scratchpad.append(failure_feedback)

                    if retry_count >= MAX_RETRIES:
                        return f"Stopped after max retries. Last error: {last_error}"

                    messages.append({"role": "user", "content": failure_feedback})
                    continue

            if recovery_mode == "syntax_write":
                is_fix_write = action in {"FILE_WRITE", "FILE_APPEND"}
                writes_target_file = False
                if is_fix_write:
                    try:
                        payload = self._parse_json_input(action_input)
                        path = str(payload.get("path", "")).strip().strip('"').strip("'")
                        writes_target_file = not recovery_file_path or path == recovery_file_path
                    except ValueError:
                        writes_target_file = False

                if not (is_fix_write and writes_target_file):
                    retry_count += 1
                    last_error = (
                        "Syntax-error recovery requires writing corrected code before rerunning."
                    )
                    failure_feedback = (
                        "The previous action failed.\n\n"
                        "Failure details:\n\n"
                        f"* Action: {action}\n"
                        f"* Input: {action_input}\n"
                        f"* Error: {last_error}\n\n"
                        "Fix the root cause instead of retrying blindly.\n"
                        f"You must write corrected code to {recovery_file_path or 'the failing file'} before running again.\n"
                        f"Your next action must be:\nAction: FILE_WRITE\nAction Input: {{\"path\": \"{recovery_file_path or 'the failing file'}\", \"content\": \"...corrected code...\"}}\n"
                        "If you execute the same command again without fixing the root cause, it will be considered a failure."
                    )
                    scratchpad.append(failure_feedback)

                    if retry_count >= MAX_RETRIES:
                        return f"Stopped after max retries. Last error: {last_error}"

                    messages.append({"role": "user", "content": failure_feedback})
                    continue

            if action == "FINAL_ANSWER":
                retry_count += 1
                last_error = "FINAL_ANSWER is not allowed here. Execute exactly one tool action and stop."
                messages.append({"role": "user", "content": last_error})
                scratchpad.append(f"Error:\n{last_error}")
                continue

            constraint_error = self._check_action_constraints(action, action_input, controls)
            if constraint_error:
                retry_count += 1
                last_error = constraint_error
                failure_feedback = (
                    "The proposed action violates task constraints.\n\n"
                    f"* Action: {action}\n"
                    f"* Input: {action_input}\n"
                    f"* Error: {constraint_error}\n\n"
                    "Choose a smaller action that directly serves the stated goal and avoids extra artifacts."
                )
                scratchpad.append(failure_feedback)
                if retry_count >= MAX_RETRIES:
                    return f"Stopped after max retries. Last error: {last_error}"
                messages.append({"role": "user", "content": failure_feedback})
                continue

            action_signature = (action, action_input)
            if action_signature == last_failed_action:
                retry_count += 1
                last_error = f"Repeated failing action blocked: {action} -> {action_input}"
                failure_feedback = (
                    "The previous action failed.\n\n"
                    "Failure details:\n\n"
                    f"* Action: {action}\n"
                    f"* Input: {action_input}\n"
                    f"* Error: {last_error}\n\n"
                    "You must analyze the cause and produce a corrected action that tries to fix the failure.\n"
                    "If the failure is in code or command syntax, fix it and rerun.\n"
                    "Do NOT repeat the same failing command.\n"
                    "Do NOT treat a failed run as a completed task.\n"
                    "If there is a pending syntax-error recovery, follow it exactly."
                )
                scratchpad.append(failure_feedback)

                if retry_count >= MAX_RETRIES:
                    return f"Stopped after max retries. Last error: {last_error}"

                messages.append({"role": "user", "content": failure_feedback})
                continue

            tool_result = self._run_action(action, action_input)

            success = tool_result.get("success", False)
            stdout = tool_result.get("stdout", "")
            stderr = tool_result.get("stderr", "")
            error_type = tool_result.get("error_type")
            observation = tool_result.get("observation", "")

            self.memory.add("Observation", observation)
            scratchpad.append(
                f"Thought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}"
            )

            if success:
                self._agent_log("SUCCESS")
                retry_count = 0
                last_error = None
                last_failed_action = None
                if recovery_mode == "syntax_read" and action == "FILE_READ":
                    recovery_mode = "syntax_write"
                elif recovery_mode == "syntax_write" and action in {"FILE_WRITE", "FILE_APPEND"}:
                    recovery_mode = None
                    recovery_file_path = None
                return observation or stdout or "Action completed successfully."

            retry_count += 1
            last_error = stderr or observation
            self._agent_log(f"FAILED: {last_error}")
            last_failed_action = action_signature
            syntax_error_detected = error_type == "syntax_error" or "SyntaxError" in last_error
            if syntax_error_detected:
                recovery_mode = "syntax_read"
                recovery_file_path = self._extract_error_file_path(last_error)
            failure_feedback = (
                "The previous action failed.\n\n"
                "Failure details:\n\n"
                f"* Action: {action}\n"
                f"* Input: {action_input}\n"
                f"* stderr: {last_error}\n\n"
                "You must analyze the cause and produce a corrected action that tries to fix the failure.\n"
                "If the failure is in code or command syntax, fix it and rerun.\n"
                "Fix the root cause instead of retrying blindly.\n"
                "Do NOT repeat the same failing command.\n"
                "Do NOT treat a failed run as a completed task."
            )
            if syntax_error_detected:
                failure_feedback += (
                    f"\nThe failing file is: {recovery_file_path or 'unknown'}\n"
                    "Recovery steps:\n"
                    f"1. FILE_READ {recovery_file_path or 'the failing file'}\n"
                    f"2. FILE_WRITE corrected code back to {recovery_file_path or 'the failing file'}\n"
                    "3. Rerun the file\n"
                    f"Your next action must be FILE_READ on {recovery_file_path or 'the failing file'}."
                )
            scratchpad.append(failure_feedback)

            if retry_count >= MAX_RETRIES:
                return f"Stopped after max retries. Last error: {last_error}"

            messages.append(
                {
                    "role": "user",
                    "content": failure_feedback,
                }
            )

        return f"Stopped after max retries. Last error: {last_error}"

    def _agent_log(self, message: str) -> None:
        log(f"[AGENT] {message}")

    def _build_prompt(self, scratchpad: list[str]) -> str:
        scratchpad_text = "\n\n".join(scratchpad)
        return f"Current session:\n{self.memory.render()}\n\nScratchpad:\n{scratchpad_text}\n"

    def _call_llm(self, messages: list[dict[str, str]], prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[*messages, {"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def _parse_output(self, text: str) -> dict[str, str]:
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Action Input:\s*([\s\S]*)", text)

        if not action_match or not input_match:
            raise ValueError(f"Could not parse model output:\n{text}")

        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip().upper()
        action_input = input_match.group(1).strip()

        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

    def _run_action(self, action: str, action_input: str) -> dict[str, Any]:
        try:
            if action not in self.tools:
                raise ValueError(f"Unknown action: {action}")

            if action in {"FILE_WRITE", "FILE_APPEND"}:
                payload = self._parse_json_input(action_input)
                result = self.tools[action].run(
                    mode="write" if action == "FILE_WRITE" else "append",
                    path=payload["path"],
                    content=payload["content"],
                )
            elif action == "FILE_READ":
                result = self.tools[action].run(mode="read", path=action_input)
            else:
                result = self.tools[action].run(action_input)

            return self._normalize_tool_result(result)
        except Exception as error:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(error),
                "error_type": None,
                "observation": f"ERROR: {error}",
            }

    def _normalize_tool_result(self, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return {
                "success": result.get("success", False),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "error_type": result.get("error_type"),
                "observation": result.get("observation", ""),
            }

        if isinstance(result, str):
            exit_code_match = re.search(r"exit_code=(\d+)", result)
            stdout_match = re.search(r"stdout:\n(.*?)(?=\nstderr:\n|\Z)", result, re.DOTALL)
            stderr_match = re.search(r"stderr:\n(.*)$", result, re.DOTALL)

            if exit_code_match:
                return {
                    "success": int(exit_code_match.group(1)) == 0,
                    "stdout": stdout_match.group(1).strip() if stdout_match else "",
                    "stderr": stderr_match.group(1).strip() if stderr_match else "",
                    "error_type": None,
                    "observation": result,
                }

            return {
                "success": not result.startswith("ERROR:"),
                "stdout": result,
                "stderr": "",
                "error_type": None,
                "observation": result,
            }

        return {
            "success": False,
            "stdout": "",
            "stderr": "Unsupported tool result type.",
            "error_type": None,
            "observation": "ERROR: Unsupported tool result type.",
        }

    def _parse_json_input(self, action_input: str) -> dict[str, Any]:
        try:
            payload = json.loads(action_input)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON action input: {error}") from error

        if not isinstance(payload, dict):
            raise ValueError("JSON action input must be an object.")
        return payload

    def _log_action(self, action: str, action_input: str) -> None:
        if action in {"FILE_WRITE", "FILE_APPEND"}:
            try:
                payload = self._parse_json_input(action_input)
                path = str(payload.get("path", "")).strip()
            except ValueError:
                path = "(unknown)"
            self._agent_log(f"Writing file: {path}")
            return

        if action == "FILE_READ":
            self._agent_log(f"Reading file: {action_input.strip()}")
            return

        if action == "SHELL":
            self._agent_log(f"Running command: {action_input.strip()}")
            return

        if action == "PYTHON":
            self._agent_log("Running Python code")
            return

    def _extract_error_file_path(self, text: str) -> str | None:
        match = re.search(r'File "([^"]+)"', text)
        if match:
            return match.group(1)
        match = re.search(r"([A-Za-z0-9_./\\-]+\.py)", text)
        return match.group(1) if match else None

    def _extract_overall_task(self, user_task: str) -> str:
        match = re.search(r"Overall task:\n(.*?)(?:\n\nProgress so far:|\Z)", user_task, re.DOTALL)
        if match:
            return match.group(1).strip()
        return user_task.strip()

    def _extract_current_step(self, user_task: str) -> str:
        match = re.search(r"Current step:\n(.*)\Z", user_task, re.DOTALL)
        if match:
            return match.group(1).strip()
        return user_task.strip()

    def _build_controls(self, overall_task: str, current_step: str) -> dict[str, Any]:
        requirements = extract_task_requirements(overall_task)
        task_lower = overall_task.lower()
        step_lower = current_step.lower()

        mentioned_files = set(re.findall(r"\b([A-Za-z0-9_.\-/]+)\b", overall_task))
        mentioned_files.update(re.findall(r"\b([A-Za-z0-9_.\-/]+)\b", current_step))

        return {
            "goal": requirements["goal"],
            "constraints": requirements["constraints"],
            "minimal_output": requirements["minimal_output"],
            "task_lower": task_lower,
            "step_lower": step_lower,
            "mentioned_files": mentioned_files,
            "allow_tests": any(word in task_lower for word in ("test", "pytest", "unittest")),
            "allow_docs": any(word in task_lower for word in ("readme", "documentation", "docs")),
            "allow_extra_features": any(
                word in task_lower for word in ("extra feature", "enhance", "improve", "refactor")
            ),
        }

    def _step_is_unnecessary(self, current_step: str, controls: dict[str, Any]) -> bool:
        step_lower = current_step.lower()
        if not controls["allow_tests"] and ("test" in step_lower or "pytest" in step_lower):
            return True
        if not controls["allow_docs"] and any(word in step_lower for word in ("readme", "documentation", "docs")):
            return True
        if not controls["allow_extra_features"] and any(
            phrase in step_lower for phrase in ("extra feature", "refactor", "cleanup", "enhance")
        ):
            return True
        return False

    def _check_action_constraints(self, action: str, action_input: str, controls: dict[str, Any]) -> str | None:
        if action not in {"FILE_WRITE", "FILE_APPEND", "SHELL"}:
            return None

        candidate_paths: list[str] = []
        if action in {"FILE_WRITE", "FILE_APPEND"}:
            try:
                payload = self._parse_json_input(action_input)
                candidate_paths.append(str(payload.get("path", "")).strip())
            except ValueError as error:
                return str(error)
        elif action == "SHELL":
            candidate_paths.extend(self._extract_paths_from_shell(action_input))

        for path in candidate_paths:
            if not path:
                continue
            violation = self._check_path_constraint(path, controls)
            if violation:
                return violation
        return None

    def _extract_paths_from_shell(self, command: str) -> list[str]:
        paths = re.findall(r"\b([A-Za-z0-9_.\-/]+\.[A-Za-z0-9_]+)\b", command)
        return [path for path in paths if not path.startswith("python")]

    def _check_path_constraint(self, path: str, controls: dict[str, Any]) -> str | None:
        normalized = Path(path).name.lower()
        if not controls["allow_tests"] and (
            normalized.startswith("test_")
            or normalized.endswith("_test.py")
            or "pytest" in normalized
        ):
            return f"Creating test artifact '{path}' is not required by the task."

        if not controls["allow_docs"] and normalized in {"readme", "readme.md", "docs", "docs.md"}:
            return f"Creating documentation artifact '{path}' is not required by the task."

        if controls["mentioned_files"]:
            referenced = {Path(item).name.lower() for item in controls["mentioned_files"] if "." in item}
            if referenced and normalized not in referenced:
                task_lower = controls["task_lower"]
                if "script" in task_lower and normalized.endswith(".py"):
                    return None
                return f"Creating unrelated file '{path}' is outside the requested output."

        return None
