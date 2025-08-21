from __future__ import annotations

import datetime as dt
import inspect
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from dotenv import dotenv_values
from pydantic import BaseModel

from llemon.apis.llm.llm_model_config import LLMModelConfig
from llemon.errors import ConfigurationError
from llemon.types import NS, History

if TYPE_CHECKING:
    from llemon.models.generate import GenerateRequest, GenerateResponse
    from llemon.models.generate_object import (
        GenerateObjectRequest,
        GenerateObjectResponse,
    )
    from llemon.models.generate_stream import (
        GenerateStreamRequest,
        GenerateStreamResponse,
    )

log = logging.getLogger(__name__)


class LLM:

    classes: ClassVar[dict[str, type[LLM]]] = {}
    configurations: ClassVar[NS] = {}
    instance: ClassVar[LLM | None] = None
    models: ClassVar[dict[str, LLMModel]] = {}

    def __init_subclass__(cls) -> None:
        cls.classes[cls.__name__] = cls
        cls.instance = None
        cls.models = {}

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"<{self}>"

    @classmethod
    def configure(cls, config_dict: NS | None = None, /, **config_kwargs: Any) -> None:
        config = dotenv_values()
        if config_dict:
            config.update(config_dict)
        if config_kwargs:
            config.update(config_kwargs)
        cls.configurations.update({key.lower(): value for key, value in config.items()})

    @classmethod
    def create(cls) -> LLM:
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
                raise ConfigurationError(f"{cls.__name__} missing configuration {parameter.name!r}")
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
                supports_streaming=supports_streaming,
                supports_json=supports_json,
                accepts_files=accepts_files,
                cost_per_1m_input_tokens=cost_per_1m_input_tokens,
                cost_per_1m_output_tokens=cost_per_1m_output_tokens,
            )
            config.load_defaults(name)
            self.models[name] = LLMModel(self, name, config)
        return self.models[name]

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        raise NotImplementedError()

    async def generate_stream(self, request: GenerateStreamRequest) -> GenerateStreamResponse:
        raise NotImplementedError()

    async def generate_object[T: BaseModel](self, request: GenerateObjectRequest[T]) -> GenerateObjectResponse[T]:
        raise NotImplementedError()

    async def prepare(self, request: GenerateRequest, state: NS) -> None:
        request.check_supported()

    async def cleanup(self, state: NS) -> None:
        pass

    def _log_history(self, history: History) -> None:
        extra = {"markup": True, "highlighter": None}
        log.debug("[bold underline]history[/]", extra=extra)
        for request, response in history:
            timestamp = f"{response.started:%H:%M:%S}-{response.ended:%H:%M:%S} ({response.duration:.2f}s)"
            log.debug(f"[bold yellow]{request.__class__.__name__}[/] [{timestamp}]", extra=extra)
            log.debug(request.format(), extra=extra)
            log.debug(response.format(), extra=extra)


from llemon.apis.llm.llm_model import LLMModel
