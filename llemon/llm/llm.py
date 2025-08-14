from __future__ import annotations

import datetime as dt
import inspect
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Self

from dotenv import dotenv_values
from pydantic import BaseModel

from ..types import Preprocessor
from ..utils import SetupError
from .llm_model import LLMModel, LLMModelConfig

if TYPE_CHECKING:
    from ..protocol import Completion, Stream, StructuredOutput, Classification, LLMOperation

log = logging.getLogger(__name__)


class LLM:

    return_incomplete_messages: ClassVar[bool] = False

    configurations: dict[str, Any] = {}
    instance: ClassVar[Self | None] = None
    models: ClassVar[dict[str, LLMModel]] = {}
    preprocessors: ClassVar[dict[str, tuple[Preprocessor, bool]]] = {}

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
    def preprocessor(cls, preprocess: Preprocessor) -> Preprocessor:
        cls.preprocessors[preprocess.__name__] = preprocess
        return preprocess
    
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
                raise SetupError(f"{cls.__name__} missing configuration {parameter.name!r}")
            kwargs[parameter.name] = value
        return cls(**kwargs)
    
    @classmethod
    def get(
        cls,
        name: str,
        *,
        knowledge_cutoff: dt.date | None = None,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        supports_multiple_responses: bool | None = None,
        supports_streaming: bool | None = None,
        supports_json: bool | None = None,
        accepts_files: list[str] | None = None,
        cost_per_1m_input_tokens: float | None = None,
        cost_per_1m_output_tokens: float | None = None,
    ) -> LLMModel:
        if not cls.instance:
            log.debug("creating instance of %s", cls.__name__)
            cls.instance = cls.create()
        self = cls.instance
        if name not in self.models:
            log.debug("creating model %s", name)
            config = LLMModelConfig(
                knowledge_cutoff=knowledge_cutoff,
                context_window=context_window,
                max_output_tokens=max_output_tokens,
                supports_multiple_responses=supports_multiple_responses,
                supports_streaming=supports_streaming,
                supports_json=supports_json,
                accepts_files=accepts_files,
                cost_per_1m_input_tokens=cost_per_1m_input_tokens,
                cost_per_1m_output_tokens=cost_per_1m_output_tokens,
            )
            config.load_defaults(name)
            self.models[name] = LLMModel(self, name, config)
        return self.models[name]
    
    async def complete(self, completion: Completion) -> None:
        raise NotImplementedError()
    
    async def stream(self, stream: Stream) -> None:
        raise NotImplementedError()

    async def construct[T: BaseModel](self, structured_output: StructuredOutput[T]) -> None:
        raise NotImplementedError()

    async def classify(self, classification: Classification) -> None:
        raise NotImplementedError()
    
    async def setup(self, operation: LLMOperation, state: dict[str, Any]) -> None:
        for name, (preprocessor, is_async) in self.preprocessors.items():
            log.debug("running preprocessor %s", name)
            if is_async:
                await preprocessor(operation)
            else:
                preprocessor(operation)
    
    async def teardown(self, state: dict[str, Any]) -> None:
        pass