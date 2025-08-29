from __future__ import annotations

import json
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, cast

from pydantic import BaseModel

import llemon.sync as llemon
from llemon.sync.types import NS, Error, FilesArgument, History, RenderArgument, ToolsArgument
from llemon.utils import schema_to_model

if TYPE_CHECKING:
    from llemon.sync import LLM


class GenerateObjectRequest[T: BaseModel](llemon.GenerateRequest):

    JSON_INSTRUCTION: ClassVar[
        str
    ] = """
    # JSON Generation Guidelines
    Answer ONLY in JSON that adheres EXACTLY to the following JSON schema:
    <SCHEMA>

    Return the ROOT object with field names EXACTLY as they appear in the JSON schema.

    # Example
    {"title": "A", "type": "object", "properties": {"foo": {"type": "string"}, "bar": {"type": "int"}}}
    Should be:
    {"foo": "hello", "bar": 1}
    Not:
    - {"Foo": "hello", "Bar": 1}
    - {"A": {"foo": "hello", "bar": 1}}
    - {"properties": {"foo": "hello", "bar": 1}}
    - [{"foo": "hello", "bar": 1}]
    """

    def __init__(
        self,
        *,
        llm: LLM,
        schema: NS | type[T],
        history: History | None = None,
        instructions: str | None = None,
        user_input: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        variants: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        prediction: str | NS | T | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            history=history,
            instructions=instructions,
            user_input=user_input,
            context=context,
            render=render,
            files=files,
            tools=tools,
            use_tool=use_tool,
            variants=variants,
            temperature=temperature,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            prediction=self._resolve_prediction(prediction),
        )
        if isinstance(schema, dict):
            self.schema = cast(type[T], schema_to_model(schema))
        else:
            self.schema = schema

    def __str__(self) -> str:
        return f"{self.llm}.generate_object({self.schema.__name__})"

    def append_json_instruction(self) -> None:
        self.append_instruction(self.JSON_INSTRUCTION.replace("<SCHEMA>", json.dumps(self.schema.model_json_schema())))

    def check_supported(self) -> None:
        super().check_supported()
        if not self.llm.config.supports_objects:
            raise Error(f"{self.llm} doesn't support object generation")

    def _resolve_prediction(self, prediction: str | NS | T | None) -> str | None:
        if prediction is None:
            return None
        if isinstance(prediction, BaseModel):
            return prediction.model_dump_json()
        try:
            return json.dumps(prediction)
        except TypeError:
            return str(prediction)


class GenerateObjectResponse[T: BaseModel](llemon.GenerateResponse):

    request: GenerateObjectRequest[T]

    def __init__(self, request: GenerateObjectRequest[T]) -> None:
        super().__init__(request)
        self._objects: list[T] = []

    def __str__(self) -> str:
        return f"{self.request.llm}: {self.object!r}"

    @cached_property
    def objects(self) -> list[T]:
        if not self.ended:
            raise self._incomplete_request()
        return self._objects

    @cached_property
    def object(self) -> T:
        return self.objects[self._selected]

    def select(self, index: int) -> None:
        super().select(index)
        self.__dict__.pop("object", None)

    def complete_object(self, *objects: T) -> None:
        self._objects = list(objects)
        super().complete_text(*[object.model_dump_json() for object in objects])
