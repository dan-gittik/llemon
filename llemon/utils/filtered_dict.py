from __future__ import annotations

from typing import Any


def filtered_dict(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value not in (None, "", [], {})}
