from __future__ import annotations

from typing import Any, ClassVar

import jinja2

from .types import FormattingArgument
from .utils import concat


class Formatting:

    format_by_default: ClassVar[bool] = True
    brackets: ClassVar[dict[str, str]] = {
        "(": ")",
        "[": "]",
        "{": "}",
        "<": ">",
    }
    default_bracket: ClassVar[str] = "{"
    formattings: ClassVar[dict[str, Formatting]] = {}

    def __init__(
        self,
        variable_start: str,
        variable_end: str,
        block_start: str,
        block_end: str,
        comment_start: str,
        comment_end: str,
    ) -> None:
        self._format = f"{variable_start}...{variable_end}"
        self._env = jinja2.Environment(
            variable_start_string=variable_start,
            variable_end_string=variable_end,
            block_start_string=block_start,
            block_end_string=block_end,
            comment_start_string=comment_start,
            comment_end_string=comment_end,
        )
    
    def __str__(self) -> str:
        return f"formatting {self._format}"
    
    def __repr__(self) -> str:
        return f"<{self!s}>"

    @classmethod
    def resolve(cls, formatting: FormattingArgument) -> Formatting | None:
        if formatting is None:
            if cls.format_by_default:
                formatting = cls.default_bracket
            else:
                return None
        if formatting is False:
            return None
        if formatting is True:
            formatting = cls.default_bracket
        if isinstance(formatting, str):
            return cls.from_bracket(formatting)
        return formatting

    @classmethod
    def from_bracket(cls, start: str) -> Formatting:
        if start not in cls.brackets:
            raise ValueError(f"Invalid bracket {start!r} (expected {concat(cls.brackets)})")
        if start not in cls.formattings:
            end = cls.brackets[start]
            cls.formattings[start] = cls(
                variable_start=start * 2,
                variable_end=end * 2,
                block_start=f"{start}%",
                block_end=f"%{end}",
                comment_start=f"{start}#",
                comment_end=f"#{end}",
            )
        return cls.formattings[start]
    
    def format(self, text: str, context_dict: dict[str, Any] | None = None, /, **context_kwargs: Any) -> str:
        context = (context_dict or {}) | context_kwargs
        template = self._env.from_string(text)
        return template.render(context)