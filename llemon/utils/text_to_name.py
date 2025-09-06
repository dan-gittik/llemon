import re

INVALID_CHARS_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def text_to_name(text: str, max_length: int = 20, default: str = "file") -> str:
    name = INVALID_CHARS_PATTERN.sub("_", text)
    name = name.strip("._-")
    if not name:
        return default
    return name[:max_length]
