from __future__ import annotations

from contextlib import asynccontextmanager
from functools import cached_property
from typing import TYPE_CHECKING, Any, AsyncIterator, Self, cast

from pydantic import BaseModel

import llemon
from llemon.types import NS, FilesArgument, HistoryArgument, RenderArgument, ToolsArgument
from llemon.utils import filtered_dict, schema_to_model

if TYPE_CHECKING:
    from llemon import (
        ClassifyResponse,
        Conversation,
        GenerateObjectResponse,
        GenerateRequest,
        GenerateResponse,
        GenerateStreamResponse,
        LLMConfig,
        LLMProvider,
        LLMTokenizer,
    )
    from llemon.objects.serializeable import DumpRefs, LoadRefs, Unpacker


class LLM(llemon.Serializeable):

    def __init__(self, provider: LLMProvider, model: str, config: LLMConfig) -> None:
        self.provider = provider
        self.model = model
        self.config = config

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"

    def __repr__(self) -> str:
        return f"<{self}>"

    @cached_property
    def tokenizer(self) -> LLMTokenizer:
        if self.config.tokenizer is None:
            tokenizer_class = llemon.LLMTokenizer
        else:
            tokenizer_class = llemon.LLMTokenizer.get_subclass(self.config.tokenizer)
        return tokenizer_class(self)

    def conversation(
        self,
        instructions: str | None = None,
        context: NS | None = None,
        tools: ToolsArgument = None,
        render: RenderArgument = True,
        history: HistoryArgument = None,
        cache: bool | None = None,
    ) -> Conversation:
        return llemon.Conversation(
            llm=self,
            instructions=instructions,
            context=context,
            tools=tools,
            history=history,
            render=render,
            cache=cache,
        )

    async def generate(
        self,
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        context: NS | None = None,
        render: RenderArgument = None,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        variants: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
        return_incomplete_message: bool | None = None,
        cache: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> GenerateResponse:
        instructions, user_input = self._resolve_messages(message1, message2)
        request = llemon.GenerateRequest(
            llm=self,
            user_input=user_input,
            instructions=instructions,
            context=context,
            render=render,
            history=history,
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
            min_p=min_p,
            top_k=top_k,
            stop=stop,
            prediction=prediction,
            return_incomplete_message=return_incomplete_message,
            cache=cache,
            timeout=timeout,
            **provider_options,
        )
        async with self._standalone(request):
            return await self.provider.generate(request)

    async def generate_stream(
        self,
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        context: NS | None = None,
        render: RenderArgument = None,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
        return_incomplete_message: bool | None = None,
        cache: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> GenerateStreamResponse:
        instructions, user_input = self._resolve_messages(message1, message2)
        request = llemon.GenerateStreamRequest(
            llm=self,
            user_input=user_input,
            instructions=instructions,
            context=context,
            render=render,
            history=history,
            files=files,
            tools=tools,
            use_tool=use_tool,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            stop=stop,
            prediction=prediction,
            return_incomplete_message=return_incomplete_message,
            cache=cache,
            timeout=timeout,
            **provider_options,
        )
        async with self._standalone(request):
            return await self.provider.generate_stream(request)

    async def generate_object[T: BaseModel](
        self,
        schema: NS | type[T],
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        context: NS | None = None,
        render: RenderArgument = None,
        history: HistoryArgument = None,
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
        min_p: float | None = None,
        top_k: int | None = None,
        prediction: str | NS | T | None = None,
        cache: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> GenerateObjectResponse[T]:
        if isinstance(schema, dict):
            model_class = cast(type[T], schema_to_model(schema))
        else:
            model_class = schema
        instructions, user_input = self._resolve_messages(message1, message2)
        request = llemon.GenerateObjectRequest(
            schema=model_class,
            llm=self,
            user_input=user_input,
            instructions=instructions,
            context=context,
            render=render,
            history=history,
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
            min_p=min_p,
            top_k=top_k,
            prediction=prediction,
            cache=cache,
            timeout=timeout,
            **provider_options,
        )
        async with self._standalone(request):
            return await self.provider.generate_object(request)

    async def classify(
        self,
        question: str,
        answers: list[str] | type[bool],
        user_input: str,
        *,
        reasoning: bool = False,
        null_answer: bool = True,
        context: NS | None = None,
        render: RenderArgument = None,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        cache: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> ClassifyResponse:
        request = llemon.ClassifyRequest(
            llm=self,
            question=question,
            answers=answers,
            user_input=user_input,
            reasoning=reasoning,
            null_answer=null_answer,
            context=context,
            render=render,
            history=history,
            files=files,
            tools=tools,
            use_tool=use_tool,
            cache=cache,
            timeout=timeout,
            **provider_options,
        )
        async with self._standalone(request):
            return await self.provider.classify(request)

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        provider = llemon.LLMProvider.get_subclass(unpacker.get("provider", str))
        config = unpacker.get("config", dict)
        config.pop("model", None)
        llm = provider.llm(model=unpacker.get("model", str), **config)
        return cast(Self, llm)

    def _dump(self, refs: DumpRefs) -> NS:
        return filtered_dict(
            provider=self.provider.__class__.__name__,
            model=self.model,
            config=self.config._dump(refs),
        )

    def _resolve_messages(self, message1: str | None, message2: str | None) -> tuple[str | None, str | None]:
        if message2 is None:
            return None, message1
        return message1, message2

    @asynccontextmanager
    async def _standalone(self, request: GenerateRequest) -> AsyncIterator[None]:
        state: NS = {}
        await self.provider.prepare_generation(request, state)
        try:
            yield
        finally:
            await self.provider.cleanup_generation(state)
