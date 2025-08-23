from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import cached_property
from typing import AsyncIterator, cast

from pydantic import BaseModel

from llemon.apis.llm.llm_tokenizer import LLMTokenizer
from llemon.apis.llm.llm_model_config import LLMModelConfig
from llemon.types import NS, FilesArgument, History, RenderArgument, ToolsArgument
from llemon.utils.schema import schema_to_model

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

    @classmethod
    def load(cls, data: NS) -> LLMModel:
        llm_class = LLM.classes[data["provider"]]
        return llm_class.model(data["name"], **(data.get("config") or {}))
    
    @cached_property
    def tokenizer(self) -> LLMTokenizer:
        return self.llm.get_tokenizer(self)

    def conversation(
        self,
        instructions: str | None = None,
        context: NS | None = None,
        tools: ToolsArgument = None,
        history: History | None = None,
        render: RenderArgument = True,
    ) -> Conversation:
        return Conversation(self, instructions, context=context, tools=tools, history=history, render=render)

    async def generate(
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
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            prediction=prediction,
            return_incomplete_message=return_incomplete_message,
        )
        async with self._standalone(request):
            return await self.llm.generate(request)

    async def generate_stream(
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
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            prediction=prediction,
            return_incomplete_message=return_incomplete_message,
        )
        async with self._standalone(request):
            return await self.llm.generate_stream(request)

    async def generate_object[T: BaseModel](
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
            top_p=top_p,
            top_k=top_k,
            prediction=prediction,
        )
        async with self._standalone(request):
            return await self.llm.generate_object(request)
    
    async def classify(
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
        async with self._standalone(request):
            return await self.llm.classify(request)

    def dump(self) -> NS:
        data: NS = dict(
            provider=self.llm.__class__.__name__,
            name=self.name,
        )
        config = self.config.dump(self.name)
        if config:
            data["config"] = config
        return data

    def _resolve_messages(self, message1: str | None, message2: str | None) -> tuple[str | None, str | None]:
        if message2 is None:
            return None, message1
        return message1, message2

    @asynccontextmanager
    async def _standalone(self, request: GenerateRequest) -> AsyncIterator[None]:
        state: NS = {}
        await self.llm.prepare(request, state)
        try:
            yield
        finally:
            await self.llm.cleanup(state)


from llemon.apis.llm.llm import LLM
from llemon.conversation import Conversation
from llemon.models.classify import ClassifyRequest, ClassifyResponse
from llemon.models.generate import GenerateRequest, GenerateResponse
from llemon.models.generate_object import GenerateObjectRequest, GenerateObjectResponse
from llemon.models.generate_stream import GenerateStreamRequest, GenerateStreamResponse
