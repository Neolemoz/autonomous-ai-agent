from pathlib import Path


class FileTool:
    def __init__(self, logger):
        self.logger = logger

    def run(self, mode: str, path: str, content: str = "") -> dict:
        path = path.strip().strip('"').strip("'")
        file_path = Path(path)

        try:
            if mode == "read":
                content = file_path.read_text()
                return {
                    "success": True,
                    "stdout": content,
                    "stderr": "",
                    "exit_code": 0,
                    "error_type": None,
                    "observation": f"File content:\n{content}",
                }

            file_path.parent.mkdir(parents=True, exist_ok=True)

            if mode == "write":
                file_path.write_text(content)
                return {
                    "success": True,
                    "stdout": f"Wrote {len(content)} bytes to {file_path}",
                    "stderr": "",
                    "exit_code": 0,
                    "error_type": None,
                    "observation": "File written successfully",
                }

            if mode == "append":
                with file_path.open("a") as handle:
                    handle.write(content)
                return {
                    "success": True,
                    "stdout": f"Appended {len(content)} bytes to {file_path}",
                    "stderr": "",
                    "exit_code": 0,
                    "error_type": None,
                    "observation": "File operation successful",
                }

            raise ValueError(f"Unsupported file mode: {mode}")
        except Exception as e:
            error_type = "file_not_found" if isinstance(e, FileNotFoundError) else "file_error"
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": None,
                "error_type": error_type,
                "observation": f"File error: {str(e)}",
            }
