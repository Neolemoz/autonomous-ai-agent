import subprocess
import sys


class PythonExecTool:
    def __init__(self, logger, safety_manager=None):
        self.logger = logger
        self.safety_manager = safety_manager

    def run(self, code: str) -> dict:
        if self.safety_manager:
            self.safety_manager.validate_python_code(code)

        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        exit_code = completed.returncode

        if exit_code == 0:
            return {
                "success": True,
                "stdout": stdout,
                "stderr": "",
                "exit_code": 0,
                "error_type": None,
                "observation": stdout,
            }

        error_type = "runtime_error"
        if "SyntaxError" in stderr:
            error_type = "syntax_error"

        return {
            "success": False,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "error_type": error_type,
            "observation": stderr or stdout,
        }
