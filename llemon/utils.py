import datetime as dt
import re
from typing import Iterator

LEADING_EMPTY_LINES = re.compile(r"^([ \t]*\r?\n)+")
INDENT_AND_CONTENT = re.compile(r"^(\s*)(.*)$", flags=re.DOTALL)


class Error(Exception):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


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