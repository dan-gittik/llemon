from __future__ import annotations

from functools import cached_property
from typing import Any, ClassVar, override

from pydantic import BaseModel

from llemon.errors import ConfigurationError
from llemon.sync.generate import GenerateRequest, GenerateResponse
from llemon.sync.types import NS
from llemon.utils.schema import schema_to_model


class GenerateObjectRequest[T: BaseModel](GenerateRequest):

    JSON_INSTRUCTION: ClassVar[str] = "Answer ONLY in JSON that adheres EXACTLY to the following JSON schema: {schema}"

    @override
    def __init__(self, *, schema: type[T], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.schema = schema

    def dump(self) -> NS:
        data = super().dump()
        data.update(
            schema=self.schema.model_json_schema(),
        )
        return data

    def append_json_instruction(self) -> None:
        self.append_instruction(self.JSON_INSTRUCTION.format(schema=self.schema.model_json_schema()))

    def check_supported(self) -> None:
        super().check_supported()
        if not self.model.config.supports_json:
            raise ConfigurationError(f"{self.model} doesn't support structured output")

    @classmethod
    def _restore(cls, data: NS) -> tuple[NS, NS]:
        args, attrs = super()._restore(data)
        args.update(
            schema=schema_to_model(data["schema"]),
        )
        return args, attrs


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

    def dump(self) -> NS:
        data = super().dump()
        data.update(
            objects=[object.model_dump() for object in self.objects],
        )
        return data

    @classmethod
    def _restore(self, data: NS) -> tuple[NS, NS]:
        args, attrs = super()._restore(data)
        request: GenerateObjectRequest[BaseModel] = args["request"]
        attrs.update(
            _objects=[request.schema.model_validate(object) for object in data["objects"]],
        )
        return args, attrs
