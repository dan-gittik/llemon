import asyncio
from concurrent.futures import Future, ThreadPoolExecutor, wait
import datetime as dt
import re
from typing import Any, Callable, Iterable

LEADING_EMPTY_LINES = re.compile(r"^([ \t]*\r?\n)+")
INDENT_AND_CONTENT = re.compile(r"^(\s*)(.*)$", flags=re.DOTALL)

executor: ThreadPoolExecutor | None = None


class Error(Exception):
    pass


class SetupError(Error):
    pass


class UnsupportedError(Error):
    pass


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
    # Skip leading empty lines, but count them to keep the line numbers correct.
    match = LEADING_EMPTY_LINES.match(text)
    if not match:
        skipped_lines = 0
    else:
        skipped_lines = match.group().count("\n")
        text = text[match.end() :]
    text = text.rstrip().expandtabs()
    indent: int | None = None
    output: list[str] = []
    for number, line in enumerate(text.splitlines(), skipped_lines):
        # First non-empty line determines the indentation to crop off.
        if indent is None:
            indent, content = split_indent(line)
            output.append(content)
            continue
        if not line.strip():
            continue
        # Subsequent lines must start with at least the same indentation.
        prefix = line[:indent]
        if prefix and not prefix.isspace():
            raise ValueError(f"expected line {number} to start with {indent!r} spaces, but got {prefix!r}")
        line = line[indent:]
        output.append(line)
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