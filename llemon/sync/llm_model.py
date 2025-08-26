from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import cached_property
from typing import Iterator, cast

from pydantic import BaseModel

from llemon.sync.types import NS, FilesArgument, History, RenderArgument, ToolsArgument
from llemon.utils import schema_to_model

log = logging.getLogger(__name__)


class LLMModel:

    def __init__(self, llm: LLM, name: str, config: LLMModelConfig) -> None:
        self.llm = llm
        self.name = name
        self.config = config

    def __str__(self) -> str:
        return f"{self.llm}/{self.name}"

    def __repr__(self) -> str:
        return f"<{self}>"

    @cached_property
    def tokenizer(self) -> LLMTokenizer:
        tokenizer_class = LLMTokenizer.get(self.config.tokenizer)
        return tokenizer_class(self)

    def conversation(
        self,
        instructions: str | None = None,
        context: NS | None = None,
        tools: ToolsArgument = None,
        history: History | None = None,
        render: RenderArgument = True,
    ) -> Conversation:
        return Conversation(self, instructions, context=context, tools=tools, history=history, render=render)

    def generate(
        self,
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        history: History | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
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
        top_k: int | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
        return_incomplete_message: bool | None = None,
    ) -> GenerateResponse:
        instructions, user_input = self._resolve_messages(message1, message2)
        request = GenerateRequest(
            model=self,
            user_input=user_input,
            instructions=instructions,
            history=history,
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
        with self._standalone(request):
            return self.llm.generate(request)

    def generate_stream(
        self,
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        history: History | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
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
        top_k: int | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
        return_incomplete_message: bool | None = None,
    ) -> GenerateStreamResponse:
        instructions, user_input = self._resolve_messages(message1, message2)
        request = GenerateStreamRequest(
            model=self,
            user_input=user_input,
            instructions=instructions,
            history=history,
            context=context,
            render=render,
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
            top_k=top_k,
            stop=stop,
            prediction=prediction,
            return_incomplete_message=return_incomplete_message,
        )
        with self._standalone(request):
            return self.llm.generate_stream(request)

    def generate_object[T: BaseModel](
        self,
        schema: type[T] | NS,
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        history: History | None = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        variants: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        prediction: T | NS | None = None,
    ) -> GenerateObjectResponse[T]:
        if isinstance(schema, dict):
            model_class = cast(type[T], schema_to_model(schema))
        else:
            model_class = schema
        instructions, user_input = self._resolve_messages(message1, message2)
        request = GenerateObjectRequest(
            schema=model_class,
            model=self,
            user_input=user_input,
            instructions=instructions,
            history=history,
            context=context,
            render=render,
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
            top_k=top_k,
            prediction=prediction,
        )
        with self._standalone(request):
            return self.llm.generate_object(request)

    def classify(
        self,
        question: str,
        answers: list[str] | type[bool],
        user_input: str,
        *,
        reasoning: bool = False,
        null_answer: bool = True,
        history: History | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
    ) -> ClassifyResponse:
        request = ClassifyRequest(
            model=self,
            question=question,
            answers=answers,
            user_input=user_input,
            reasoning=reasoning,
            null_answer=null_answer,
            history=history,
            context=context,
            render=render,
            files=files,
            tools=tools,
            use_tool=use_tool,
        )
        with self._standalone(request):
            return self.llm.classify(request)

    def _resolve_messages(self, message1: str | None, message2: str | None) -> tuple[str | None, str | None]:
        if message2 is None:
            return None, message1
        return message1, message2

    @contextmanager
    def _standalone(self, request: GenerateRequest) -> Iterator[None]:
        state: NS = {}
        self.llm.prepare(request, state)
        try:
            yield
        finally:
            self.llm.cleanup(state)


from llemon.sync.conversation import Conversation
from llemon.sync.llm import LLM
from llemon.genai.llm_model_config import LLMModelConfig
from llemon.sync.llm_tokenizer import LLMTokenizer
from llemon.sync.classify import ClassifyRequest, ClassifyResponse
from llemon.sync.generate import GenerateRequest, GenerateResponse
from llemon.sync.generate_object import GenerateObjectRequest, GenerateObjectResponse
from llemon.sync.generate_stream import GenerateStreamRequest, GenerateStreamResponse
