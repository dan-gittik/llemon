from __future__ import annotations

import re
from functools import cached_property
from typing import Any, Callable, ClassVar

from jinja2 import Environment, StrictUndefined, pass_context
from jinja2.runtime import Context

from llemon.sync.types import RenderArgument
from llemon.utils.concat import concat
from llemon.utils.parallelize import parallelize, to_sync


class Rendering:

    render_by_default: ClassVar[bool] = True
    closing_brackets: ClassVar[dict[str, str]] = {
        "(": ")",
        "[": "]",
        "{": "}",
        "<": ">",
    }
    default_bracket: ClassVar[str] = "{"
    renderers: ClassVar[dict[str, Rendering]] = {}
    namespace: ClassVar[dict[str, Any]] = {}
    predicates: ClassVar[dict[str, Callable]] = {}

    def __init__(self, bracket: str) -> None:
        self.bracket = bracket
        open, close = self.bracket * 2, self.closing_bracket * 2
        self._env = Environment(
            variable_start_string=open,
            variable_end_string=close,
            block_start_string=f"{self.bracket}%",
            block_end_string=f"%{self.closing_bracket}",
            comment_start_string=f"{self.bracket}#",
            comment_end_string=f"#{self.closing_bracket}",
            undefined=StrictUndefined,
        )
        self._env.tests.update(self.predicates)
        self._env.globals.update(self.namespace)
        self._regex = re.compile(rf"{re.escape(open)}\s*!\s*(.*?){re.escape(close)}")

    def __str__(self) -> str:
        return f"rendering of {self.bracket}...{self.closing_bracket}"

    def __repr__(self) -> str:
        return f"<{self}>"

    @classmethod
    def resolve(cls, render: RenderArgument) -> Rendering | None:
        if render is None:
            if cls.render_by_default:
                render = cls.default_bracket
            else:
                return None
        if render is False:
            return None
        if render is True:
            render = cls.default_bracket
        if isinstance(render, str):
            return cls.from_bracket(render)
        return render

    @classmethod
    def from_bracket(cls, bracket: str) -> Rendering:
        if bracket not in cls.closing_brackets:
            raise ValueError(f"Invalid bracket {bracket!r} (expected {concat(cls.closing_brackets)})")
        if bracket not in cls.renderers:
            cls.renderers[bracket] = cls(bracket)
        return cls.renderers[bracket]

    @classmethod
    def function(cls, function: Callable[..., Any]) -> Callable[..., Any]:
        cls.namespace[function.__name__] = function
        return function

    @classmethod
    def predicate(cls, function: Callable[..., Any]) -> Callable[..., Any]:
        function = to_sync(function)

        @pass_context
        def test(ctx: Context, *args: Any, **kwargs: Any) -> Any:
            return function(dict(ctx), *args, **kwargs)

        cls.predicates[function.__name__] = test
        return function

    @cached_property
    def closing_bracket(self) -> str:
        return self.closing_brackets[self.bracket]

    def render(self, text: str, context_dict: dict[str, Any] | None = None, /, **context_kwargs: Any) -> str:
        context = (context_dict or {}) | context_kwargs
        if matches := list(self._regex.finditer(text)):
            ctx = {key: to_sync(value) if callable(value) else value for key, value in context.items()}
            expressions = [match.group(1) for match in matches]
            evaluations = parallelize((self._evaluate, exp, ctx) for exp in expressions)
            output: list[str] = []
            offset = 0
            for match, evaluation in zip(matches, evaluations):
                output.append(text[offset : match.start()])
                output.append(str(evaluation))
                offset = match.end()
            output.append(text[offset:])
            text = "".join(output)
        template = self._env.from_string(text)
        return template.render(context)

    def _evaluate(self, expression: str, context: dict[str, Any]) -> str:
        return eval(expression, context)
