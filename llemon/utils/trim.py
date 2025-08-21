import re

INDENT_AND_CONTENT = re.compile(r"^(\s*)(.*)$", flags=re.DOTALL)


def split_indent(text: str) -> tuple[int, str]:
    if match := INDENT_AND_CONTENT.match(text):
        whitespace, content = match.groups()
        return len(whitespace), content
    return 0, text


def trim(text: str) -> str:
    first_indent: int | None = None
    trimmed: list[str] = []
    for line in text.rstrip().expandtabs().splitlines():
        if not line.strip():
            continue
        if first_indent is None:
            first_indent, content = split_indent(line)
            trimmed.append(content)
            continue
        indent, content = split_indent(line)
        trimmed.append(line[min(indent, first_indent) :])
    return "\n".join(trimmed)
