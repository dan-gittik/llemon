from __future__ import annotations

import json
import logging
import traceback
from functools import cached_property
from typing import TYPE_CHECKING, Any, Self, cast

from pydantic import BaseModel

import llemon.sync as llemon
from llemon.sync.types import NS, Error
from llemon.utils import Emoji, filtered_dict, to_sync, undefined

if TYPE_CHECKING:
    from llemon.sync import Tool
    from llemon.sync.serializeable import DumpRefs, LoadRefs, Unpacker

log = logging.getLogger(__name__)


class Call(llemon.Serializeable):

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
        log.debug(Emoji.TOOL + "%s", self.signature)
        try:
            self._return_value = to_sync(self.tool.function)(**self.arguments)
            log.debug(Emoji.TOOL + "%s returned %r", self.signature, self._return_value)
        except Exception as error:
            self._error = self._format_error(error)
            log.debug(Emoji.TOOL + "%s raised %r", self.signature, self._error)

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        tool = cast(llemon.Tool, refs.get_tool(unpacker.get("tool", str)))
        result = unpacker.get("result", dict)
        return cls(
            id=unpacker.get("id", str),
            tool=tool,
            arguments=unpacker.get("arguments", dict),
            return_value=result.get("return_value", undefined),
            error=result.get("error"),
        )

    def _dump(self, refs: DumpRefs) -> NS:
        refs.add_tool(self.tool)
        return filtered_dict(
            id=self.id,
            tool=self.tool.name,
            arguments=self.arguments,
            result=self.result,
        )

    def _incomplete_call(self) -> Error:
        return Error(f"{self} didn't run yet")

    def _format_error(self, error: Exception) -> str:
        return "".join(traceback.format_exception(error.__class__, error, error.__traceback__))
