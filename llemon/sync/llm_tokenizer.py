from __future__ import annotations

from typing import ClassVar, Sequence

from pydantic import BaseModel

from llemon.sync.llm_model import LLMModel
from llemon.sync.generate import GenerateRequest
from llemon.sync.generate_object import GenerateObjectRequest
from llemon.sync.types import NS, Error, FilesArgument, RenderArgument, ToolsArgument
from llemon.utils import concat, schema_to_model


class LLMTokenizer:

    label: ClassVar[str] = ""
    classes: ClassVar[dict[str, type[LLMTokenizer]]] = {}

    def __init__(self, model: LLMModel) -> None:
        self.model = model

    def __str__(self) -> str:
        return f"{self.model} tokenizer"

    def __repr__(self) -> str:
        return f"<{self}>"

    def __init_subclass__(cls) -> None:
        if not cls.label:
            raise TypeError(f"{cls.__name__} must define a label")
        if cls.label in cls.classes:
            raise TypeError(f"{cls.__name__} label {cls.label!r} is already used by {cls.classes[cls.label].__name__}")
        cls.classes[cls.label] = cls

    @classmethod
    def get(cls, label: str | None = None) -> type[LLMTokenizer]:
        if label is None:
            return LLMTokenizer
        if label not in cls.classes:
            raise ValueError(f"no tokenizer {label!r} (available tokenizers are {concat(cls.classes)})")
        return cls.classes[label]

    def count(
        self,
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument | None = None,
        files: FilesArgument | None = None,
        tools: ToolsArgument | None = None,
        schema: type[BaseModel] | NS | None = None,
    ) -> int:
        instructions, user_input = self.model._resolve_messages(message1, message2)
        args: NS = dict(
            model=self.model,
            user_input=user_input,
            instructions=instructions,
            context=context,
            render=render,
            files=files,
            tools=tools,
        )
        if schema is not None:
            if isinstance(schema, dict):
                schema = schema_to_model(schema)
            return self._count(GenerateObjectRequest[BaseModel](schema=schema, **args))
        else:
            return self._count(GenerateRequest(**args))

    def encode(self, *texts: str) -> list[int]:
        raise self._unsupported()

    def decode(self, *ids: int) -> str:
        raise self._unsupported()

    def parse(self, text: str) -> Sequence[LLMToken]:
        raise self._unsupported()

    def _count(self, request: GenerateRequest) -> int:
        return self.model.llm.count_tokens(request)

    def _unsupported(self) -> Error:
        return Error(f"{self.model} tokenizer does not support this operation")


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
