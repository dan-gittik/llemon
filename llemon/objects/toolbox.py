from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Self

import llemon
from llemon.types import NS
from llemon.utils import Superclass, filtered_dict, parse_parameters

if TYPE_CHECKING:
    from llemon import Tool
    from llemon.objects.serializeable import DumpRefs, LoadRefs, Unpacker


class Toolbox(Superclass, llemon.Serializeable):

    tool_suffix: ClassVar[str] = "_tool"
    description_suffix: ClassVar[str] = "_description"
    render_prefix: ClassVar[str] = "render_"

    def __init__(self, name: str) -> None:
        self.name = name
        self._suffix = llemon.Tool.suffix()
        self._init: dict[str, Any] = {}

    @property
    def tool_names(self) -> list[str]:
        tool_names = []
        for attribute in dir(self):
            if attribute.endswith(self.tool_suffix):
                tool_names.append(attribute.removesuffix(self.tool_suffix))
        return tool_names

    @property
    def render_names(self) -> list[str]:
        render_names = []
        for attribute in dir(self):
            if attribute.startswith(self.render_prefix):
                render_names.append(attribute.removeprefix(self.render_prefix))
        return render_names

    @cached_property
    def tools(self) -> list[Tool]:
        tools: list[Tool] = []
        for name in self.tool_names:
            function: Callable[..., Any] = getattr(self, f"{name}{self.tool_suffix}")
            get_description = getattr(self, f"{name}{self.description_suffix}", None)
            if get_description:
                description = get_description()
            else:
                description = function.__doc__ or ""
            parameters = parse_parameters(function)
            tool = llemon.Tool(f"{self.name}.{name}{self._suffix}", description, parameters, function, self)
            tools.append(tool)
        return tools

    @cached_property
    def renders(self) -> dict[str, Callable[..., Any]]:
        renders: dict[str, Callable[..., Any]] = {}
        for name in self.render_names:
            function = getattr(self, f"{self.render_prefix}{name}")
            renders[name] = function
        return renders

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        toolbox_class = cls.get_subclass(unpacker.get("type", str))
        toolbox = toolbox_class(**unpacker.get("init", dict))
        toolbox._suffix = unpacker.get("suffix", str)
        return toolbox

    def _dump(self, refs: DumpRefs) -> NS:
        return filtered_dict(
            type=self.__class__.__name__,
            name=self.name,
            init=self._init,
            suffix=self._suffix,
        )
