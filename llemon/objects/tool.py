from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable, NoReturn, Self

import llemon
from llemon.types import NS, ToolsArgument
from llemon.utils import filtered_dict, parse_parameters, random_suffix, trim

if TYPE_CHECKING:
    from llemon import Toolbox
    from llemon.objects.serializeable import DumpRefs, LoadRefs, Unpacker


class Tool(llemon.Serializeable):

    def __init__(
        self,
        name: str,
        description: str,
        parameters: NS,
        function: Callable[..., Any] | None = None,
        toolbox: Toolbox | None = None,
    ) -> None:
        if function is None:
            function = self._not_runnable
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.toolbox = toolbox
        self.display_name = self.name.rsplit("__", 1)[0]
        self.compatible_name = self.name.rsplit(".", 1)[-1]

    def __str__(self) -> str:
        return f"tool {self.display_name!r}"

    def __repr__(self) -> str:
        return f"<{self}>"

    @classmethod
    def resolve(cls, tools: ToolsArgument) -> list[Tool | Toolbox]:
        if tools is None:
            return []
        resolved: list[Tool | Toolbox] = []
        for tool in tools:
            if callable(tool):
                resolved.append(cls.from_function(tool))
            else:
                resolved.append(tool)
        return resolved

    @classmethod
    def from_function(cls, function: Callable[..., Any]) -> Tool:
        parameters = parse_parameters(function)
        return cls(
            name=function.__name__ + random_suffix(),
            description=trim(function.__doc__ or ""),
            parameters=parameters,
            function=function,
        )

    @classmethod
    def _not_runnable(self, *args, **kwargs: Any) -> NoReturn:
        raise RuntimeError(f"{self} is not associated with a runnable function")

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        function_name = unpacker.get("function", str, None)
        if function_name:
            module_name, function_name = function_name.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                function = getattr(module, function_name)
            except ImportError:
                raise ValueError(f"unable to load {unpacker}.function: module {module_name} not found")
            except AttributeError:
                raise ValueError(
                    f"unable to load {unpacker}.function: module {module_name} has no function {function_name}"
                )
        else:
            function = None
        return cls(
            name=unpacker.get("name", str),
            description=unpacker.get("description", str, ""),
            parameters=unpacker.get("parameters", dict),
            function=function,
        )

    def _dump(self, refs: DumpRefs) -> NS:
        data = filtered_dict(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
        if self.function is not self._not_runnable:
            data["function"] = f"{self.function.__module__}.{self.function.__name__}"
        return data
