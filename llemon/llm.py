from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, ClassVar, Self, overload

from dotenv import dotenv_values
from pydantic import BaseModel

from .model import Model

if TYPE_CHECKING:
    from .requests import (
        CompletionRequest,
        CompletionResponse,
        StreamResponse,
        StructuredOutputResponse,
        ClassificationRequest,
        ClassificationResponse,
    )


class LLM:

    configurations: dict[str, Any] = {}
    instance: ClassVar[Self | None] = None
    models: ClassVar[dict[str, Model]] = {}

    def __init_subclass__(cls) -> None:
        cls.instance = None
        cls.models = {}
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}"
    
    def __repr__(self) -> str:
        return f"<{self}>"
    
    @classmethod
    def configure(cls, config_dict: dict[str, Any] | None = None, /, **config_kwargs: Any) -> None:
        config = dotenv_values()
        if config_dict:
            config.update(config_dict)
        if config_kwargs:
            config.update(config_kwargs)
        cls.configurations.update({key.lower(): value for key, value in config.items()})
    
    @classmethod
    def create(cls) -> Self:
        if cls.__init__ is object.__init__:
            return cls()
        if not cls.configurations:
            cls.configure()
        signature = inspect.signature(cls.__init__)
        parameters = list(signature.parameters.values())[1:]  # skip self
        kwargs = {}
        prefix = cls.__name__.lower()
        for parameter in parameters:
            name = f"{prefix}_{parameter.name}"
            if name in cls.configurations:
                value = cls.configurations[name]
            elif parameter.default is not parameter.empty:
                value = parameter.default
            else:
                raise ValueError(f"{cls.__name__} missing configuration {parameter.name!r}")
            kwargs[parameter.name] = value
        return cls(**kwargs)
    
    @overload
    @classmethod
    def get(cls, model: str) -> Model: ...
    
    @overload
    @classmethod
    def get(cls, model: None) -> Self: ...

    @classmethod
    def get(cls, model: str | None = None) -> Model | Self:
        if not cls.instance:
            cls.instance = cls.create()
        if model:
            return cls.instance.model(model)
        return cls.instance
    
    def model(self, name: str) -> Model:
        if name not in self.models:
            self.models[name] = Model(self, name)
        return self.models[name]
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        raise NotImplementedError()
    
    async def stream(self, request: CompletionRequest) -> StreamResponse:
        raise NotImplementedError()

    async def construct[T: BaseModel](self, schema: type[T], request: CompletionRequest) -> StructuredOutputResponse[T]:
        raise NotImplementedError()

    async def classify(self, request: ClassificationRequest) -> ClassificationResponse:
        raise NotImplementedError()