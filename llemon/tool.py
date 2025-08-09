from __future__ import annotations

import asyncio
import concurrent.futures
import json
import inspect
from functools import cached_property
from typing import Any, Callable, ClassVar, NoReturn, get_type_hints

from pydantic import BaseModel, ConfigDict

from .types import CallArgument, CallMessage, ToolsArgument, ToolSchema

schemas: dict[Callable[..., Any], ToolSchema] = {}
undefined = object()


class Tool:

    def __init__(self, function: Callable[..., Any]) -> None:
        self.function = function
        self.name = function.__name__
        self.description = function.__doc__ or ""
        self.schema = self._parse_schema()
    
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
        return {function.__name__: Tool(function) for function in tools}
    
    def _parse_schema(self) -> ToolSchema:
        if self.function in schemas:
            return schemas[self.function]
        annotations = get_type_hints(self.function)
        annotations.pop("return", None)
        model_class: type[BaseModel] = type(self.name, (BaseModel,), {
            "__annotations__": annotations,
            "model_config": ConfigDict(extra='forbid')
        })
        schema: ToolSchema = {
            "name": self.name,
            "description": self.description,
            "parameters": model_class.model_json_schema(),
        }
        schemas[self.function] = schema
        return schema


class ToolSchema(Tool):

    def __init__(self, schema: ToolSchema) -> None:
        self.schema = schema
        self.name = schema["name"]
        self.function = self._not_runnable
    
    def _not_runnable(self, *args, **kwargs: Any) -> NoReturn:
        raise RuntimeError(f"{self} only contains a schema and isn't directly runnable")


class Call:

    executor: ClassVar[concurrent.futures.Executor | None] = None

    def __init__(self, id: str, tool: Tool, arguments: str | dict[str, Any], return_value: Any = undefined) -> None:
        self.id = id
        self.tool = tool
        if isinstance(arguments, str):
            self.arguments_json = arguments
            self.arguments = json.loads(arguments)
        else:
            self.arguments_json = json.dumps(arguments)
            self.arguments = arguments
        self._return_value = return_value
    
    def __str__(self) -> str:
        output = f"call {self.id!r}: {self.tool.name}({self.arguments_json!r})"
        if self._return_value is not undefined:
            output += f" -> {self.return_value_string}"
        return output
    
    def __repr__(self) -> str:
        return f"<{self}>"
    
    @classmethod
    def get_executor(cls) -> concurrent.futures.Executor:
        if cls.executor is None:
            cls.executor = concurrent.futures.ThreadPoolExecutor()
        return cls.executor
    
    @classmethod
    def resolve(cls, call: CallArgument) -> Call:
        if isinstance(call, Call):
            return call
        try:
            return_value = json.loads(call["return_value"])
        except json.JSONDecodeError:
            return_value = call["return_value"]
        return cls(
            id=call["id"],
            tool=ToolSchema(call["tool"]),
            arguments_json=call["arguments"],
            return_value=return_value,
        )
    
    @classmethod
    def run_all(cls, calls: list[Call]) -> None:
        executor = cls.get_executor()
        futures: list[concurrent.futures.Future[Any]] = []
        for call in calls:
            future = executor.submit(call.run)
            futures.append(future)
        concurrent.futures.wait(futures)

    @classmethod
    async def async_run_all(cls, calls: list[Call]) -> None:
        tasks = [asyncio.create_task(call.async_run()) for call in calls]
        await asyncio.gather(*tasks, return_exceptions=True)

    @cached_property
    def return_value(self) -> Any:
        if self._return_value is undefined:
            raise self._didnt_run()
        return self._return_value
    
    @cached_property
    def return_value_string(self) -> str:
        if isinstance(self.return_value, BaseModel):
            return self.return_value.model_dump_json()
        try:
            return json.dumps(self.return_value)
        except TypeError:
            return str(self.return_value)

    def run(self) -> None:
        try:
            output = self.tool.function(**self.arguments)
            self._return_value = output
        except Exception as error:
            self._return_value = {"error": str(error)}

    async def async_run(self) -> None:
        try:
            if inspect.iscoroutinefunction(self.tool.function):
                output = await self.tool.function(**self.arguments)
            else:
                output = await asyncio.to_thread(self.tool.function, **self.arguments)
            self._return_value = output
        except Exception as error:
            self._return_value = {"error": str(error)}
    
    def to_message(self) -> CallMessage:
        return {
            "role": "call",
            "id": self.id,
            "tool": self.tool.schema,
            "arguments": self.arguments_json,
            "return_value": self.return_value_string,
        }

    def _didnt_run(self) -> RuntimeError:
        return RuntimeError(f"{self} didn't run yet")