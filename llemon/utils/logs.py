from __future__ import annotations

import logging

from rich.logging import RichHandler


class Emoji:
    SYSTEM = "💡 "
    USER = "🧑 "
    ASSISTANT = "🤖 "
    FILE = "📎  "
    TOOL = "🛠️  "
    EMBED = "🧩 "
    TRANSCRIBE = "✒️  "
    SYNTHESIZE = "🎤 "


def enable_logs(level: int = logging.DEBUG) -> None:
    class Logger(logging.Logger):

        def __init__(self, name: str) -> None:
            super().__init__(name)
            if name.startswith(__package__.split(".")[0]):
                handler = RichHandler(rich_tracebacks=True)
                handler.setLevel(level)
                self.propagate = False
                self.setLevel(level)
                if not any(isinstance(handler, RichHandler) for handler in self.handlers):
                    self.addHandler(handler)

    logging.setLoggerClass(Logger)
