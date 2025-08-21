import logging

from rich.logging import RichHandler

SYSTEM = "💡 "
USER = "🧑 "
ASSISTANT = "🤖 "
FILE = "📎  "
TOOL = "🛠️  "


def enable_logs(level: int = logging.DEBUG) -> None:
    handler = RichHandler(rich_tracebacks=True)
    handler.setLevel(level)
    for name in logging.root.manager.loggerDict:
        if not name.startswith(__package__.split(".")[0]):
            continue
        log = logging.getLogger(name)
        log.propagate = False
        log.setLevel(level)
        if not any(isinstance(handler, RichHandler) for handler in log.handlers):
            log.addHandler(handler)
