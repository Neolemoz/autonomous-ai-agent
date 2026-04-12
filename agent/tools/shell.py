import subprocess


class ShellTool:
    def __init__(self, safety_manager, logger):
        self.safety_manager = safety_manager
        self.logger = logger

    def run(self, command: str) -> dict:
        command = command.strip().strip('"').strip("'")
        self.safety_manager.validate_shell_command(command)

        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        exit_code = completed.returncode

        error_type = None
        if "not found" in stderr:
            error_type = "command_not_found"
        elif "SyntaxError" in stderr:
            error_type = "syntax_error"
        elif "No such file" in stderr:
            error_type = "file_not_found"

        return {
            "success": exit_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "error_type": error_type,
            "observation": f"exit_code={exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        }
