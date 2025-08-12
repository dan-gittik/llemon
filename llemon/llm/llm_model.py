from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, cast, overload

from pydantic import BaseModel

from ..conversation import Conversation
from ..protocol import Classification, Completion, LLMOperation, Stream, StructuredOutput
from ..schema import schema_to_model
from ..types import FilesArgument, HistoryArgument, ToolsArgument, FormattingArgument
from .model_config import LLMModelConfig

if TYPE_CHECKING:
    from .llm import LLM

log = logging.getLogger(__name__)


class LLMModel:

    def __init__(self, llm: LLM, name: str, config: LLMModelConfig) -> None:
        self.llm = llm
        self.name = name
        self.config = config
    
    def __str__(self) -> str:
        return f"{self.llm} {self.name!r}"
    
    def __repr__(self) -> str:
        return f"<{self}>"
        
    def __call__(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        tools: ToolsArgument = None,
        history: HistoryArgument = None,
        formatting: FormattingArgument = True,
    ) -> Conversation:
        return Conversation(self, prompt, context=context, tools=tools, history=history, formatting=formatting)

    @overload
    async def complete(
        self,
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        num_responses: None,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
    ) -> str: ...

    @overload
    async def complete(
        self,
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        num_responses: int,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
    ) -> list[str]: ...

    async def complete(
        self,
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        num_responses: int | None = None,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
    ) -> str | list[str]:
        system_prompt, user_message = self._resolve_messages(message1, message2)
        completion = Completion(
            model=self,
            user_message=user_message,
            system_prompt=system_prompt,
            history=history,
            files=files,
            tools=tools,
            use_tool=use_tool,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            num_responses=num_responses,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            prediction=prediction,
        )
        log.debug("completing %s", completion)
        async with self._operation(completion):
            await self.llm.complete(completion)
        if num_responses is None:
            return completion.text
        return completion.texts

    async def stream(
        self,
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
    ) -> AsyncIterator[str]:
        system_prompt, user_message = self._resolve_messages(message1, message2)
        stream = Stream(
            model=self,
            user_message=user_message,
            system_prompt=system_prompt,
            history=history,
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
            prediction=prediction
        )
        log.debug("streaming %s", stream)
        async with self._operation(stream):
            await self.llm.stream(stream)
            async for chunk in stream:
                yield chunk
    
    @overload
    async def construct(
        self,
        schema: dict[str, Any],
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        num_responses: None,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        prediction: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    @overload
    async def construct[T: BaseModel](
        self,
        schema: type[T],
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        num_responses: None,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        prediction: T | None = None,
    ) -> T: ...

    @overload
    async def construct(
        self,
        schema: dict[str, Any],
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        num_responses: int,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        prediction: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    @overload
    async def construct[T: BaseModel](
        self,
        schema: type[T],
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        num_responses: int,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        prediction: T | None = None,
    ) -> list[T]: ...

    async def construct[T: BaseModel](
        self,
        schema: type[T] | dict[str, Any],
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        num_responses: int | None = None,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        prediction: T | dict[str, Any] | None = None,
    ) -> T | dict[str, Any] | list[T] | list[dict[str, Any]]:
        if isinstance(schema, dict):
            model_class = cast(type[T], schema_to_model(schema))
            return_model = False
        else:
            model_class = schema
            return_model = True
        system_prompt, user_message = self._resolve_messages(message1, message2)
        structured_output = StructuredOutput(
            model=self,
            schema=model_class,
            user_message=user_message,
            system_prompt=system_prompt,
            history=history,
            files=files,
            temperature=temperature,
            seed=seed,
            num_responses=num_responses,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            top_k=top_k,
            tools=tools,
            use_tool=use_tool,
            prediction=prediction,
        )
        log.debug("generating structured output %s", structured_output)
        async with self._operation(structured_output):
            await self.llm.construct(structured_output)
        if num_responses is None:
            if return_model:
                log.debug("returning single %s", model_class.__name__)
                return structured_output.object
            log.debug("returning single dict")
            return structured_output.dict
        if return_model:
            log.debug("returning list of %s", model_class.__name__)
            return structured_output.objects
        log.debug("returning list of dicts")
        return structured_output.dicts

    async def classify(
        self,
        question: str,
        answers: list[str],
        message: str | None = None,
        *,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
    ) -> str:
        classification = Classification(
            model=self,
            user_message=message,
            question=question,
            answers=answers,
            history=history,
            files=files,
            tools=tools,
            use_tool=use_tool,
        )
        log.debug("classifying %s", classification)
        async with self._operation(classification):
            await self.llm.classify(classification)
        return classification.answer

    def _resolve_messages(self, message1: str | None, message2: str | None) -> tuple[str | None, str | None]:
        if message2 is None:
            return None, message1
        return message1, message2
    
    @asynccontextmanager
    async def _operation(self, operation: LLMOperation) -> AsyncIterator[None]:
        state = {}
        await self.llm.setup(operation, state)
        try:
            yield
        finally:
            await self.llm.teardown(state)


class LLMModelGetter:

    def __init__(self, name: str) -> None:
        self.name = name
    
    def __get__(self, instance: LLM, owner: type[LLM]) -> LLMModel:
        return owner.get(self.name)