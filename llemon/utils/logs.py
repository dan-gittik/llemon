from __future__ import annotations

import logging

from rich.logging import RichHandler

LOG_NAME_PREFIX = __package__.split(".")[0]


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
            if self.name.startswith(LOG_NAME_PREFIX):
                add_rich_handler_to_llemon_logger(self, level)

    logging.setLoggerClass(Logger)
    for name in logging.root.manager.loggerDict:
        if name.startswith(LOG_NAME_PREFIX):
            logger = logging.getLogger(name)
            add_rich_handler_to_llemon_logger(logger, level)


def add_rich_handler_to_llemon_logger(logger: logging.Logger, level: int) -> None:
    logger.setLevel(level)
    logger.propagate = False
    handler = RichHandler(rich_tracebacks=True)
    handler.setLevel(level)
    if not any(isinstance(handler, RichHandler) for handler in logger.handlers):
        logger.addHandler(handler)