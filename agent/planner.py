import json
import os
import re
from typing import Any

from openai import OpenAI


PLANNER_PROMPT = """You are a task planner for an autonomous coding agent.

Produce a strict JSON object with this shape:
{
  "goal": "...",
  "constraints": ["..."],
  "minimal_output": "...",
  "steps": ["...", "..."]
}

Rules:
- Keep the plan minimal.
- Extract the real goal from the task.
- Extract explicit and implicit constraints.
- Include constraints that forbid unnecessary artifacts such as tests, README files, docs, refactors, or extra features unless the task explicitly asks for them.
- "minimal_output" must describe the smallest acceptable deliverable.
- Use at most 5 steps.
- Each step must be exactly one action-level task for the existing agent.
- Each step must map to one tool call such as FILE_WRITE, FILE_READ, SHELL, or PYTHON.
- Use concrete file names when files are required.
- Respect the requested mode if present.
- Do not explain anything outside the JSON object.
"""


def _extract_mode(task: str) -> tuple[str, str]:
    match = re.match(r"\[MODE:\s*([a-zA-Z_]+)\]\s*(?:\n|$)", task.strip())
    if not match:
        return "general", task.strip()
    mode = match.group(1).strip().lower()
    cleaned_task = re.sub(r"^\[MODE:\s*[a-zA-Z_]+\]\s*\n?", "", task.strip(), count=1)
    return mode, cleaned_task.strip()


def _fallback_plan(task: str) -> dict[str, Any]:
    mode, cleaned_task = _extract_mode(task)
    constraints = [
        "Do only the minimum work required by the task.",
        "Do not create tests, README files, docs, or extra files unless explicitly requested.",
    ]
    if mode == "coder":
        constraints.extend(
            [
                "Only generate code needed for the requested task.",
                "Do not create tests, README files, or extra support files.",
            ]
        )
    elif mode == "tester":
        constraints.extend(
            [
                "Only generate or update tests.",
                "Do not modify implementation files unless explicitly requested.",
            ]
        )
    elif mode == "planner":
        constraints.extend(
            [
                "Only produce a plan.",
                "Do not generate code, tests, or documentation artifacts.",
            ]
        )

    return {
        "goal": cleaned_task,
        "constraints": constraints,
        "minimal_output": "Only the exact requested artifact or result.",
        "steps": [cleaned_task],
    }


def _mentions_any(text: str, words: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def extract_task_requirements(task: str) -> dict[str, Any]:
    mode, cleaned_task = _extract_mode(task)
    lowered = cleaned_task.lower()
    constraints: list[str] = []

    for match in re.findall(r"\b(?:do not|don't|without|no)\b([^.;\n]+)", cleaned_task, re.IGNORECASE):
        constraint = match.strip(" .")
        if constraint:
            constraints.append(constraint)

    if " only " in f" {lowered} ":
        constraints.append("Only do what is explicitly requested.")

    if mode == "coder":
        constraints.append("Only generate code.")
        constraints.append("Do not create test files.")
        constraints.append("Do not create README or documentation files.")
        constraints.append("Do not add extra features or unrelated refactors.")
    elif mode == "tester":
        constraints.append("Only generate tests.")
        constraints.append("Do not create README or documentation files.")
        constraints.append("Do not add implementation features beyond what tests require.")
    elif mode == "planner":
        constraints.append("Only plan steps.")
        constraints.append("Do not create files.")
        constraints.append("Do not create tests, README files, or documentation files.")
    else:
        if not _mentions_any(lowered, ("test", "pytest", "unittest")):
            constraints.append("Do not create test files.")
        if not _mentions_any(lowered, ("readme", "documentation", "docs")):
            constraints.append("Do not create README or documentation files.")
        if not _mentions_any(lowered, ("refactor", "cleanup", "improve", "enhance", "extra feature")):
            constraints.append("Do not add extra features or unrelated refactors.")

    minimal_output = "Only the exact requested artifact or output."
    if mode == "planner":
        minimal_output = "A minimal plan only."
    elif "script" in lowered or ".py" in lowered:
        minimal_output = "One working Python file with only the requested behavior."
    elif "file" in lowered:
        minimal_output = "Only the requested file with the requested content."
    elif mode == "tester":
        minimal_output = "Only the requested test file or test updates."

    seen: set[str] = set()
    normalized_constraints: list[str] = []
    for item in constraints:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            normalized_constraints.append(item)

    return {
        "goal": cleaned_task,
        "constraints": normalized_constraints,
        "minimal_output": minimal_output,
        "mode": mode,
    }


def _normalize_plan(data: Any, task: str) -> dict[str, Any]:
    fallback = _fallback_plan(task)
    requirements = extract_task_requirements(task)
    if not isinstance(data, dict):
        return fallback

    goal = str(data.get("goal") or fallback["goal"]).strip()
    minimal_output = str(data.get("minimal_output") or fallback["minimal_output"]).strip()
    constraints_raw = data.get("constraints", fallback["constraints"])
    steps_raw = data.get("steps", fallback["steps"])

    constraints = (
        [str(item).strip() for item in constraints_raw if str(item).strip()]
        if isinstance(constraints_raw, list)
        else list(fallback["constraints"])
    )
    steps = (
        [str(item).strip() for item in steps_raw if str(item).strip()]
        if isinstance(steps_raw, list)
        else list(fallback["steps"])
    )

    if requirements["mode"] == "planner":
        steps = ["Produce the minimal plan only."]
    elif requirements["mode"] == "tester":
        steps = [
            step
            for step in steps
            if any(word in step.lower() for word in ("test", "assert", "pytest", "unittest"))
        ] or steps[:1]
    elif requirements["mode"] == "coder":
        steps = [
            step
            for step in steps
            if not any(word in step.lower() for word in ("test", "readme", "documentation", "docs"))
        ]

    if not steps:
        steps = list(fallback["steps"])

    return {
        "goal": goal,
        "constraints": constraints,
        "minimal_output": minimal_output,
        "steps": steps[:5],
    }


def parse_plan_payload(plan_text: str, task: str | None = None) -> dict[str, Any]:
    default_task = task or plan_text
    try:
        return _normalize_plan(json.loads(plan_text), default_task)
    except json.JSONDecodeError:
        raw_steps = [match.strip() for match in re.findall(r"^\d+\.\s*(.+)$", plan_text, re.MULTILINE)]
        fallback = _fallback_plan(default_task)
        if raw_steps:
            fallback["steps"] = raw_steps
        fallback.update(extract_task_requirements(default_task))
        if fallback.get("mode") == "planner":
            fallback["steps"] = ["Produce the minimal plan only."]
        elif fallback.get("mode") == "coder":
            fallback["steps"] = [
                step
                for step in fallback["steps"]
                if not any(word in step.lower() for word in ("test", "readme", "documentation", "docs"))
            ] or fallback["steps"]
        elif fallback.get("mode") == "tester":
            fallback["steps"] = [
                step
                for step in fallback["steps"]
                if any(word in step.lower() for word in ("test", "assert", "pytest", "unittest"))
            ] or fallback["steps"]
        return fallback


def generate_plan(task: str) -> str:
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    mode, cleaned_task = _extract_mode(task)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": f"Mode: {mode}\nTask: {cleaned_task}"},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    return json.dumps(parse_plan_payload(content, task), indent=2)


def replan(task: str, old_plan: str, scratchpad: str) -> str:
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    mode, cleaned_task = _extract_mode(task)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": PLANNER_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "You are replanning a failed task.\n\n"
                    f"Mode: {mode}\n\n"
                    f"Task:\n{cleaned_task}\n\n"
                    f"Previous plan:\n{old_plan}\n\n"
                    f"Execution log:\n{scratchpad}\n\n"
                    "Generate a new improved plan that fixes the failure while staying minimal."
                ),
            },
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    return json.dumps(parse_plan_payload(content, task), indent=2)


def parse_plan(plan_text: str) -> list[str]:
    payload = parse_plan_payload(plan_text)
    raw_steps = payload["steps"]
    steps: list[str] = []
    split_pattern = re.compile(r"\s+and\s+", re.IGNORECASE)
    action_markers = ("create ", "write ", "read ", "run ", "execute ")

    for step in raw_steps:
        lowered = step.lower()
        if " and " in lowered and sum(marker in lowered for marker in action_markers) >= 2:
            parts = [part.strip(" .") for part in split_pattern.split(step) if part.strip(" .")]
            steps.extend(parts)
            continue
        steps.append(step)

    return steps
