from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, Future, wait
import json
import inspect
from functools import cached_property
import logging
import traceback
from typing import Any, Callable, ClassVar, NoReturn, get_type_hints

from pydantic import BaseModel, ConfigDict

from .tooling import Toolbox
from .types import CallArgument, CallMessage, ToolsArgument, ToolSchema
from .utils import TOOL, trim

log = logging.getLogger(__name__)
schemas: dict[Callable[..., Any], ToolSchema] = {}
undefined = object()


class Tool:

    def __init__(self, schema: ToolSchema, function: Callable[..., Any] | None = None) -> None:
        if function is None:
            function = self._not_runnable
        self.schema = schema
        self.function = function
    
    def __str__(self) -> str:
        return f"tool {self.name!r}"
    
    def __repr__(self) -> str:
        return f"<{self}>"

    @classmethod
    def resolve(cls, tools: ToolsArgument) -> dict[str, Tool]:
        if tools is None:
            return {}
        if isinstance(tools, dict):
            return tools
        resolved: dict[str, Tool] = {}
        for tool in tools:
            if isinstance(tool, Toolbox):
                for name, description, function in tool.tools:
                    schema = cls._parse_schema(function)
                    schema.update(name=name, description=trim(description))
                    resolved[name] = cls(schema, function)
            else:
                resolved[tool.__name__] = cls.from_function(tool)
        return resolved
    
    @classmethod
    def from_function(cls, function: Callable[..., Any]) -> Tool:
        return cls(cls._parse_schema(function), function)

    
    @classmethod
    def _not_runnable(self, *args, **kwargs: Any) -> NoReturn:
        raise RuntimeError(f"{self} is not associated with a runnable function")
    
    @classmethod
    def _parse_schema(self, function: Callable[..., Any]) -> ToolSchema:
        if function in schemas:
            return schemas[function]
        annotations = get_type_hints(function)
        annotations.pop("return", None)
        model_class: type[BaseModel] = type(function.__name__, (BaseModel,), {
            "__annotations__": annotations,
            "model_config": ConfigDict(extra='forbid')
        })
        schema = ToolSchema(
            name=function.__name__,
            description=trim(function.__doc__ or ""),
            parameters=model_class.model_json_schema(),
        )
        schemas[function] = schema
        return schema
    
    @property
    def name(self) -> str:
        return self.schema["name"]
    
    @property
    def description(self) -> str:
        return self.schema["description"]


class Call:

    executor: ClassVar[ThreadPoolExecutor | None] = None

    def __init__(
        self,
        id: str,
        tool: Tool,
        arguments: dict[str, Any],
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
    
    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        if cls.executor is None:
            cls.executor = ThreadPoolExecutor()
        return cls.executor
    
    @classmethod
    def resolve(cls, call: CallArgument) -> Call:
        if isinstance(call, Call):
            return call
        result = json.loads(call["result"])
        return_value = result.get("return_value", undefined)
        error = result.get("error")
        return cls(
            id=call["id"],
            tool=Tool(call["tool"]),
            arguments_json=call["arguments"],
            return_value=return_value,
            error=error,
        )
    
    @classmethod
    def run_all(cls, calls: list[Call]) -> None:
        executor = cls.get_executor()
        futures: list[Future[Any]] = []
        for call in calls:
            future = executor.submit(call.run)
            futures.append(future)
        wait(futures)

    @classmethod
    async def async_run_all(cls, calls: list[Call]) -> None:
        tasks = [asyncio.create_task(call.async_run()) for call in calls]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    @cached_property
    def signature(self) -> str:
        args = f", ".join(f"{key}={value!r}" for key, value in self.arguments.items())
        return f"{self.tool.name}({args})"
    
    @cached_property
    def arguments_json(self) -> str:
        return json.dumps(self.arguments)

    @cached_property
    def return_value(self) -> Any:
        if self._error:
            raise self._error
        if self._return_value is undefined:
            raise self._didnt_run()
        return self._return_value
    
    @cached_property
    def error(self) -> str | None:
        if not self._error and self._return_value is undefined:
            raise self._didnt_run()
        return self._error
    
    @cached_property
    def result(self) -> dict[str, Any]:
        if self._error:
            return {"error": self.error}
        elif self._return_value is undefined:
            raise self._didnt_run()
        return {"return_value": self.return_value}
    
    @cached_property
    def result_json(self) -> str:
        result: dict[str, Any] = {}
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
        log.debug("running %s", self.signature)
        try:
            self._return_value = self.tool.function(**self.arguments)
            log.debug("%s returned %r", self.signature, self._return_value)
        except Exception as error:
            self._error = self._format_error(error)
            log.debug("%s raised %r", self.signature, self._error)

    async def async_run(self) -> None:
        log.debug(TOOL + "%s", self.signature)
        try:
            if inspect.iscoroutinefunction(self.tool.function):
                return_value = await self.tool.function(**self.arguments)
            else:
                return_value = await asyncio.to_thread(self.tool.function, **self.arguments)
            log.debug("%s returned %r", self.signature, return_value)
            self._return_value = return_value
        except Exception as error:
            self._error = self._format_error(error)
            log.debug("%s raised %r", self.signature, self._error)
    
    def to_message(self) -> CallMessage:
        return CallMessage(
            id=self.id,
            tool=self.tool.schema,
            arguments=self.arguments_json,
            result=self.result_json,
        )

    def _didnt_run(self) -> RuntimeError:
        return RuntimeError(f"{self} didn't run yet")

    def _format_error(self, error: Exception) -> str:
        return "".join(traceback.format_exception(error.__class__, error, error.__traceback__))