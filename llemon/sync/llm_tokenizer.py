from __future__ import annotations

import json
from typing import TYPE_CHECKING, Sequence

from pydantic import BaseModel

import llemon.sync as llemon
from llemon.sync.types import NS, Error, FilesArgument, RenderArgument, ToolsArgument
from llemon.utils import Superclass, schema_to_model

if TYPE_CHECKING:
    from llemon.sync import LLM, GenerateRequest


class LLMTokenizer(Superclass):

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def __str__(self) -> str:
        return f"{self.llm} tokenizer"

    def __repr__(self) -> str:
        return f"<{self}>"

    def count(
        self,
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        context: NS | None = None,
        render: RenderArgument | None = None,
        files: FilesArgument | None = None,
        tools: ToolsArgument | None = None,
        schema: type[BaseModel] | NS | None = None,
    ) -> int:
        instructions, user_input = self.llm._resolve_messages(message1, message2)
        args: NS = dict(
            llm=self.llm,
            user_input=user_input,
            instructions=instructions,
            context=context,
            render=render,
            files=files,
            tools=tools,
        )
        request: GenerateRequest
        if schema is not None:
            if isinstance(schema, dict):
                schema = schema_to_model(schema)
            request = llemon.GenerateObjectRequest[BaseModel](schema=schema, **args)
        else:
            request = llemon.GenerateRequest(**args)
        if self._count is not LLMTokenizer._count:
            return self._count_tokens(request)
        return self.llm.provider.count_tokens(request)

    def encode(self, *texts: str) -> list[int]:
        raise self._unsupported()

    def decode(self, *ids: int) -> str:
        raise self._unsupported()

    def parse(self, text: str) -> Sequence[LLMToken]:
        raise self._unsupported()

    def _count_tokens(self, request: GenerateRequest) -> int:
        total = 0
        if request.instructions:
            if isinstance(request, llemon.GenerateObjectRequest) and not request.llm.config.supports_structured_output:
                request.append_json_instruction()
            total += self._count(request.render_instructions()) + 3
        for request_, response_ in request.history:
            if isinstance(request_, llemon.GenerateRequest):
                total += self._count(request_.user_input) + 3
            if isinstance(response_, llemon.GenerateResponse):
                total += self._count(response_.text) + 3
        if request.user_input:
            total += self._count(request.user_input) + 3
        tools: list[dict] = []
        for name, tool in request.tools_dict.items():
            tools.append(
                dict(
                    type="function",
                    function=dict(
                        name=name,
                        description=tool.description,
                        parameters=tool.parameters,
                    ),
                )
            )
        if tools:
            total += self._count(json.dumps(tools, separators=(",", ":"))) + 3
        if isinstance(request, llemon.GenerateObjectRequest):
            schema = request.schema.model_json_schema()
            total += self._count(json.dumps(schema, separators=(",", ":"))) + 3
        return total + 3

    def _count(self, text: str) -> int:
        raise NotImplementedError()

    def _unsupported(self) -> Error:
        return Error(f"{self.llm} tokenizer does not support this operation")


class LLMToken:

    def __init__(self, id: int) -> None:
        self.id = id

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"<token {self.id}: {self.text}>"

    @property
    def text(self) -> str:
        raise NotImplementedError()

    @property
    def offset(self) -> int:
        raise NotImplementedError()
