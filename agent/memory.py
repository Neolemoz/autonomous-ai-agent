class AgentMemory:
    def __init__(self):
        self.entries = []

    def add(self, label: str, content: str):
        self.entries.append((label, str(content)))

    def render(self) -> str:
        return "\n".join(f"{label}: {content}" for label, content in self.entries)
