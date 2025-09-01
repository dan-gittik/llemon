from __future__ import annotations

import datetime as dt
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, cast

from pydantic import BaseModel

import llemon.sync as llemon
from llemon.sync.types import NS, History

if TYPE_CHECKING:
    from llemon.sync import (
        LLM,
        ClassifyRequest,
        ClassifyResponse,
        GenerateObjectRequest,
        GenerateObjectResponse,
        GenerateRequest,
        GenerateResponse,
        GenerateStreamRequest,
        GenerateStreamResponse,
    )

log = logging.getLogger(__name__)


class LLMProvider(ABC, llemon.Provider):

    llms: ClassVar[dict[str, LLM]] = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.llms = {}

    @classmethod
    def llm(
        cls,
        model: str,
        *,
        tokenizer: str | None = None,
        knowledge_cutoff: dt.date | None = None,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        unsupported_parameters: list[str] | None = None,
        supports_streaming: bool | None = None,
        supports_structured_output: bool | None = None,
        supports_json: bool | None = None,
        supports_tools: bool | None = None,
        supports_logit_biasing: bool | None = None,
        accepts_files: list[str] | None = None,
        cost_per_1m_input_tokens: float | None = None,
        cost_per_1m_cache_tokens: float | None = None,
        cost_per_1m_output_tokens: float | None = None,
    ) -> LLM:
        self = cls.get()
        if model not in self.llms:
            log.debug("creating model %s", model)
            config = llemon.LLMConfig(
                model=model,
                tokenizer=tokenizer,
                knowledge_cutoff=knowledge_cutoff,
                context_window=context_window,
                max_output_tokens=max_output_tokens,
                unsupported_parameters=unsupported_parameters,
                supports_streaming=supports_streaming,
                supports_structured_output=supports_structured_output,
                supports_json=supports_json,
                supports_tools=supports_tools,
                supports_logit_biasing=supports_logit_biasing,
                accepts_files=accepts_files,
                cost_per_1m_input_tokens=cost_per_1m_input_tokens,
                cost_per_1m_cache_tokens=cost_per_1m_cache_tokens,
                cost_per_1m_output_tokens=cost_per_1m_output_tokens,
            )
            self.llms[model] = llemon.LLM(self, model, config)
        return self.llms[model]

    @abstractmethod
    def count_tokens(self, request: GenerateRequest) -> int:
        raise NotImplementedError()

    @abstractmethod
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        raise NotImplementedError()

    @abstractmethod
    def generate_stream(self, request: GenerateStreamRequest) -> GenerateStreamResponse:
        raise NotImplementedError()

    @abstractmethod
    def generate_object[T: BaseModel](self, request: GenerateObjectRequest[T]) -> GenerateObjectResponse[T]:
        raise NotImplementedError()

    def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        response = llemon.ClassifyResponse(request)
        reasoning: str | None = None
        if request.llm.config.supports_objects:
            log.debug("classifying with structured output")
            generate_object_request = request.to_object_request()
            generate_object_response = self.generate_object(generate_object_request)
            data = generate_object_response.object.model_dump()
            answer_num = cast(int, data["answer"])
            reasoning = data.get("reasoning")
        else:
            log.debug("classifying with generated text")
            generate_response = self.generate(request)
            if not generate_response.text.isdigit():
                raise request.error(f"{request} answer was not a number: {generate_response.text}")
            answer_num = int(generate_response.text)
        if not 0 <= answer_num < len(request.answers):
            raise request.error(f"{request} answer number was out of range: {answer_num}")
        answer = request.answers[answer_num]
        log.debug("classification: %s (%s)", answer, reasoning or "no reasoning")
        response.complete_answer(answer, reasoning)
        return response

    def prepare_generation(self, request: GenerateRequest, state: NS) -> None:
        request.check_supported()

    def cleanup_generation(self, state: NS) -> None:
        pass

    def _log_history(self, history: History) -> None:
        extra = {"markup": True, "highlighter": None}
        log.debug("[bold underline]history[/]", extra=extra)
        for request, response in history:
            timestamp = f"{response.started:%H:%M:%S}-{response.ended:%H:%M:%S} ({response.duration:.2f}s)"
            log.debug(f"[bold yellow]{request.__class__.__name__}[/] [{timestamp}]", extra=extra)
            log.debug(request.format(), extra=extra)
            log.debug(response.format(), extra=extra)
