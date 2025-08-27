from __future__ import annotations

import json
import logging
import secrets
import traceback
from functools import cached_property
from typing import Any, Callable, ClassVar, NoReturn, get_type_hints

from pydantic import BaseModel, ConfigDict

from llemon.sync.types import NS, Error, ToolsArgument
from llemon.utils import TOOL, Superclass, to_sync, trim

log = logging.getLogger(__name__)
PARAMETER_SCHEMAS: dict[Callable[..., Any], NS] = {}
undefined = object()


class Tool:

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
    def from_function(cls, function: Callable[..., Any]) -> Tool:
        parameters = _parse_parameters(function)
        return cls(
            name=function.__name__ + _suffix(),
            description=trim(function.__doc__ or ""),
            parameters=parameters,
            function=function,
        )

    @classmethod
    def _not_runnable(self, *args, **kwargs: Any) -> NoReturn:
        raise RuntimeError(f"{self} is not associated with a runnable function")


class Call:

    def __init__(
        self,
        id: str,
        tool: Tool,
        arguments: NS,
        return_value: Any = undefined,
        error: str | None = None,
    ) -> None:
        self.id = id
        self.tool = tool
        self.arguments = arguments
        self._return_value = return_value
        self._error = error

    def __str__(self) -> str:
        output = [f"call {self.id!r}: {self.signature}"]
        if self._return_value is not undefined:
            output.append(f" -> {self._return_value}")
        elif self._error is not None:
            output.append(f" -> {self._error!r}")
        return "".join(output)

    def __repr__(self) -> str:
        return f"<{self}>"

    @cached_property
    def signature(self) -> str:
        args = ", ".join(f"{key}={value!r}" for key, value in self.arguments.items())
        return f"{self.tool.display_name}({args})"

    @cached_property
    def arguments_json(self) -> str:
        return json.dumps(self.arguments)

    @cached_property
    def return_value(self) -> Any:
        if self._error:
            raise Error(self._error)
        if self._return_value is undefined:
            raise self._incomplete_call()
        return self._return_value

    @cached_property
    def error(self) -> str | None:
        if not self._error and self._return_value is undefined:
            raise self._incomplete_call()
        return self._error

    @cached_property
    def result(self) -> NS:
        if self._error:
            return {"error": self.error}
        elif self._return_value is undefined:
            raise self._incomplete_call()
        return {"return_value": self.return_value}

    @cached_property
    def result_json(self) -> str:
        result: NS = {}
        if self._error:
            result["error"] = self._error
        else:
            if isinstance(self.return_value, BaseModel):
                return_value = self.return_value.model_dump_json()
            try:
                return_value = json.dumps(self.return_value)
            except TypeError:
                return_value = str(self.return_value)
            result["return_value"] = return_value
        return json.dumps(result)

    def run(self) -> None:
        log.debug(TOOL + "%s", self.signature)
        try:
            self._return_value = to_sync(self.tool.function)(**self.arguments)
            log.debug(TOOL + "%s returned %r", self.signature, self._return_value)
        except Exception as error:
            self._error = self._format_error(error)
            log.debug(TOOL + "%s raised %r", self.signature, self._error)

    def _incomplete_call(self) -> Error:
        return Error(f"{self} didn't run yet")

    def _format_error(self, error: Exception) -> str:
        return "".join(traceback.format_exception(error.__class__, error, error.__traceback__))


class Toolbox(Superclass):

    tool_suffix: ClassVar[str] = "_tool"
    description_suffix: ClassVar[str] = "_description"
    render_prefix: ClassVar[str] = "render_"

    def __init__(self, name: str) -> None:
        self.name = name
        self._suffix = _suffix()
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
            parameters = _parse_parameters(function)
            tool = Tool(f"{self.name}.{name}{self._suffix}", description, parameters, function, self)
            tools.append(tool)
        return tools

    @cached_property
    def renders(self) -> dict[str, Callable[..., Any]]:
        renders: dict[str, Callable[..., Any]] = {}
        for name in self.render_names:
            function = getattr(self, f"{self.render_prefix}{name}")
            renders[name] = function
        return renders


def resolve_tools(tools: ToolsArgument) -> list[Tool | Toolbox]:
    if tools is None:
        return []
    resolved: list[Tool | Toolbox] = []
    for tool in tools:
        if callable(tool):
            resolved.append(Tool.from_function(tool))
        else:
            resolved.append(tool)
    return resolved


def _parse_parameters(function: Callable[..., Any]) -> NS:
    if function in PARAMETER_SCHEMAS:
        return PARAMETER_SCHEMAS[function]
    annotations = get_type_hints(function)
    annotations.pop("return", None)
    model_class: type[BaseModel] = type(
        function.__name__,
        (BaseModel,),
        {"__annotations__": annotations, "model_config": ConfigDict(extra="forbid")},
    )
    schema = model_class.model_json_schema()
    PARAMETER_SCHEMAS[function] = schema
    return schema


def _suffix() -> str:
    return f"__{secrets.token_hex(8)}"
