import asyncio
from concurrent.futures import Future, ThreadPoolExecutor, wait
import datetime as dt
import logging
import re
from typing import Any, Callable, Iterable

from rich.logging import RichHandler

USER = "🧑 "
ASSISTANT = "🤖 "
FILE = "📎  "
TOOL = "🛠️  "
INDENT_AND_CONTENT = re.compile(r"^(\s*)(.*)$", flags=re.DOTALL)

executor: ThreadPoolExecutor | None = None


class Error(Exception):
    pass


class SetupError(Error):
    pass


class UnsupportedError(Error):
    pass


def enable_logs(level: int = logging.DEBUG) -> None:
    handler = RichHandler(rich_tracebacks=True)
    handler.setLevel(level)
    for name in logging.root.manager.loggerDict:
        if not name.startswith(__package__):
            continue
        log = logging.getLogger(name)
        log.propagate = False
        log.setLevel(level)
        if not any(isinstance(handler, RichHandler) for handler in log.handlers):
            log.addHandler(handler)


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def concat(iterable: Iterable[Any], conjunction: str = "or") -> str:
    items = list(iterable)
    if not items:
        return "<none>"
    if len(items) == 1:
        return str(items[0])
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return ", ".join(map(str, items[:-1])) + f" {conjunction} {items[-1]}"


def split_indent(text: str) -> tuple[int, str]:
    # Regex is guaranteed to match, so we ignore the type check to avoid unreachable code.
    whitespace, content = INDENT_AND_CONTENT.match(text).groups()  # type: ignore
    indent = len(whitespace)
    return indent, content


def trim(text: str) -> str:
    text = text.rstrip().expandtabs()
    first_indent: int | None = None
    output: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if first_indent is None:
            first_indent, content = split_indent(line)
            output.append(content)
            continue
        indent, content = split_indent(line)
        output.append(line[min(indent, first_indent):])
    return "\n".join(output)


async def async_parallelize(calls: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]]) -> list[Any]:
    tasks = [asyncio.create_task(call(*args, **kwargs)) for call, args, kwargs in calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors: list[Exception] = []
    failed: list[str] = []
    for result, (call, args, kwargs) in zip(results, calls):
        if isinstance(result, Exception):
            errors.append(result)
            params = ", ".join([str(arg) for arg in args] + [f"{key}={value!r}" for key, value in kwargs.items()])
            failed.append(f"{call.__name__}({params})")
    if errors:
        raise ExceptionGroup(f"failed to run {concat(failed, 'and')}", errors)
    return results


async def parallelize(calls: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]]) -> list[Any]:
    if executor is None:
        executor = ThreadPoolExecutor()
    futures: list[Future[Any]] = []
    for call, args, kwargs in calls:
        future = executor.submit(call, *args, **kwargs)
        futures.append(future)
    wait(futures)
    results: list[Any] = []
    errors: list[Exception] = []
    failed: list[str] = []
    for future, (call, args, kwargs) in zip(futures, calls):
        try:
            result = future.result()
            results.append(result)
        except Exception as error:
            errors.append(error)
            params = ", ".join([str(arg) for arg in args] + [f"{key}={value!r}" for key, value in kwargs.items()])
            failed.append(f"{call.__name__}({params})")
    if errors:
        raise ExceptionGroup(f"failed to run {concat(failed, 'and')}", errors)
    return results