from typing import Any, Iterable


def concat(iterable: Iterable[Any], conjunction: str = "or") -> str:
    items = list(iterable)
    if not items:
        return "<none>"
    if len(items) == 1:
        return str(items[0])
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return ", ".join(map(str, items[:-1])) + f" {conjunction} {items[-1]}"
