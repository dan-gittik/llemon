from __future__ import annotations

from functools import cached_property
from typing import ClassVar

from pydantic import BaseModel

from llemon.sync.generate import GenerateRequest, GenerateResponse
from llemon.sync.types import NS, Error, FilesArgument, History, RenderArgument, ToolsArgument


class GenerateObjectRequest[T: BaseModel](GenerateRequest):

    JSON_INSTRUCTION: ClassVar[str] = "Answer ONLY in JSON that adheres EXACTLY to the following JSON schema: {schema}"

    def __init__(
        self,
        *,
        model: LLMModel,
        schema: type[T],
        history: History | None = None,
        instructions: str | None = None,
        user_input: str | None = None,
        context: NS | None = None,
        render: RenderArgument | None = None,
        files: FilesArgument = None,
        tools: ToolsArgument | None = None,
        use_tool: bool | str | None = None,
        variants: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        prediction: str | NS | BaseModel | None = None,
        return_incomplete_message: bool | None = None,
    ) -> None:
        super().__init__(
            model=model,
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
            max_tokens=max_tokens,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            prediction=prediction,
            return_incomplete_message=return_incomplete_message,
        )
        self.schema = schema

    def append_json_instruction(self) -> None:
        self.append_instruction(self.JSON_INSTRUCTION.format(schema=self.schema.model_json_schema()))

    def check_supported(self) -> None:
        super().check_supported()
        if not self.model.config.supports_json:
            raise Error(f"{self.model} doesn't support structured output")


class GenerateObjectResponse[T: BaseModel](GenerateResponse):

    request: GenerateObjectRequest[T]

    def __init__(self, request: GenerateObjectRequest[T]) -> None:
        super().__init__(request)
        self._objects: list[T] = []

    def __str__(self) -> str:
        return f"{self.request.model}: {self.object}"

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


from llemon.sync.llm_model import LLMModel