from __future__ import annotations

import datetime as dt
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator, ClassVar, cast

from pydantic import BaseModel

import llemon
from llemon.types import NS, History

if TYPE_CHECKING:
    from llemon import (
        LLM,
        ClassifyRequest,
        ClassifyResponse,
        GenerateObjectRequest,
        GenerateObjectResponse,
        GenerateRequest,
        GenerateResponse,
        GenerateStreamResponse,
    )

log = logging.getLogger(__name__)


class LLMProvider(ABC, llemon.Provider):

    llm_models: ClassVar[dict[str, LLM]] = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.llm_models = {}

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
        if model not in self.llm_models:
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
            self.llm_models[model] = llemon.LLM(self, model, config)
        return self.llm_models[model]

    @abstractmethod
    async def count_tokens(self, request: GenerateRequest) -> int:
        raise NotImplementedError()

    async def prepare_generation(self, request: GenerateRequest) -> None:
        pass

    async def cleanup_generation(self, state: NS) -> None:
        pass

    async def generate(self, request: GenerateRequest, stream: bool | None = None) -> GenerateResponse:
        async with self._generation(request):
            if stream:
                response = llemon.GenerateResponse(request)
                await self._generate(request, response)
                return response
            else:
                response = llemon.GenerateStreamResponse(request)
                await self._generate_stream(request, response)
                return response

    async def generate_object[T: BaseModel](self, request: GenerateObjectRequest[T]) -> GenerateObjectResponse[T]:
        async with self._generation(request):
            response = llemon.GenerateObjectResponse(request)
            await self._generate_object(request, response)
            return response

    async def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        async with self._generation(request):
            response = llemon.ClassifyResponse(request)
            await self._classify(request, response)
            return response

    @abstractmethod
    async def _generate(self, request: GenerateRequest, response: GenerateResponse) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def _generate_stream(self, request: GenerateRequest, response: GenerateStreamResponse) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def _generate_object[T: BaseModel](
        self,
        request: GenerateObjectRequest[T],
        response: GenerateObjectResponse[T],
    ) -> None:
        raise NotImplementedError()

    async def _classify(self, request: ClassifyRequest, response: ClassifyResponse) -> None:
        reasoning: str | None = None
        if request.llm.config.supports_objects:
            log.debug("classifying with structured output")
            generate_object_request = request.to_object_request()
            generate_object_response = llemon.GenerateObjectResponse(generate_object_request)
            await self._generate_object(generate_object_request, generate_object_response)
            data = generate_object_response.object.model_dump()
            answer_num = cast(int, data["answer"])
            reasoning = data.get("reasoning")
        else:
            log.debug("classifying with generated text")
            generate_response = llemon.GenerateResponse(request)
            await self._generate(request, generate_response)
            if not generate_response.text.isdigit():
                raise request.error(f"{request} answer was not a number: {generate_response.text}")
            answer_num = int(generate_response.text)
        if not 0 <= answer_num < len(request.answers):
            raise request.error(f"{request} answer number was out of range: {answer_num}")
        answer = request.answers[answer_num]
        log.debug("classification: %s (%s)", answer, reasoning or "no reasoning")
        response.complete_answer(answer, reasoning)

    @asynccontextmanager
    async def _generation(self, request: GenerateRequest) -> AsyncIterator[None]:
        request.check_supported()
        await self.prepare_generation(request)
        try:
            yield
        finally:
            if request.cleanup:
                await self.cleanup_generation(request.state)

    def _log_history(self, history: History) -> None:
        extra = {"markup": True, "highlighter": None}
        log.debug("[bold underline]history[/]", extra=extra)
        for request, response in history:
            timestamp = f"{response.started:%H:%M:%S}-{response.ended:%H:%M:%S} ({response.duration:.2f}s)"
            log.debug(f"[bold yellow]{request.__class__.__name__}[/] [{timestamp}]", extra=extra)
            log.debug(request.format(), extra=extra)
            log.debug(response.format(), extra=extra)
