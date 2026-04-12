class SafetyManager:
    def __init__(self):
        self.blocked_patterns = ["rm -rf", "shutdown", "reboot"]
        self.blocked_python_patterns = [
            "import os",
            "import shutil",
            "import subprocess",
            "os.remove",
            "os.rmdir",
            "shutil.rmtree",
        ]
        self.allowed_prefixes = [
            "ls",
            "pwd",
            "echo",
            "cat",
            "python",
            "python3",
            "pip",
            "mkdir",
            "touch",
            "head",
            "tail",
            "grep",
            "find",
            "sed",
        ]

    def validate_shell_command(self, command: str):
        normalized = command.strip().lower()

        for pattern in self.blocked_patterns:
            if pattern in normalized:
                raise ValueError(f"Blocked dangerous command: {pattern}")

        if not any(normalized.startswith(prefix) for prefix in self.allowed_prefixes):
            raise ValueError(f"Command not in allowlist: {command}")

    def validate_python_code(self, code: str):
        normalized = code.strip().lower()

        for pattern in self.blocked_python_patterns:
            if pattern in normalized:
                raise ValueError(f"Blocked dangerous Python pattern: {pattern}")
