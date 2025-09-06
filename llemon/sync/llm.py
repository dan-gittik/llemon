from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Self, cast, overload

from pydantic import BaseModel

import llemon.sync as llemon
from llemon.sync.types import NS, FilesArgument, HistoryArgument, RenderArgument, ToolsArgument
from llemon.utils import filtered_dict, schema_to_model

if TYPE_CHECKING:
    from llemon.sync import (
        ClassifyResponse,
        Conversation,
        GenerateObjectResponse,
        GenerateResponse,
        GenerateStreamResponse,
        LLMConfig,
        LLMProvider,
        LLMTokenizer,
    )
    from llemon.sync.serializeable import DumpRefs, LoadRefs, Unpacker


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

    @overload
    def generate(
        self,
        user_input: str | None = None,
        *,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        variants: int | None = None,
        stream: Literal[False] | None = None,
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
    ) -> GenerateResponse: ...

    @overload
    def generate(
        self,
        user_input: str | None = None,
        *,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        variants: int | None = None,
        stream: Literal[True],
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
    ) -> GenerateStreamResponse: ...

    def generate(
        self,
        user_input: str | None = None,
        *,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        variants: int | None = None,
        stream: bool | None = None,
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
    ) -> GenerateResponse | GenerateStreamResponse:
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
            stream=stream,
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
        return self.provider.generate(request, stream=stream)

    def generate_object[T: BaseModel](
        self,
        schema: NS | type[T],
        user_input: str | None = None,
        *,
        instructions: str | None = None,
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
        return self.provider.generate_object(request)

    def classify(
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
        return self.provider.classify(request)

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        provider = llemon.LLMProvider.get_subclass(unpacker.get("provider", str))
        llm = provider.llm(model=unpacker.get("model", str), **unpacker.get("config", dict, {}))
        return cast(Self, llm)

    def _dump(self, refs: DumpRefs) -> NS:
        config = self.config._dump(refs)
        del config["model"]
        return filtered_dict(
            provider=self.provider.__class__.__name__,
            model=self.model,
            config=config,
        )


LLMModel = llemon.Model[LLM]
