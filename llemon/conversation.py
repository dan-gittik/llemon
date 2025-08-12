from __future__ import annotations

from contextlib import asynccontextmanager
import json
from types import TracebackType
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Self, overload, cast

from pydantic import BaseModel

from .interaction import History, Interaction
from .formatting import Formatting
from .protocol import Completion, Classification, Stream, StructuredOutput, LLMOperation
from .schema import schema_to_model
from .tool import Tool
from .types import FormattingArgument, FilesArgument, HistoryArgument, Messages, SystemMessage, ToolsArgument
from .utils import now

if TYPE_CHECKING:
    from .llm import LLM, LLMModel


class Conversation:

    def __init__(
        self,
        model: LLMModel,
        prompt: str,
        context: dict[str, Any] | None = None,
        history: History | None = None,
        tools: ToolsArgument = None,
        formatting: FormattingArgument = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.context = context if context is not None else {}
        self.history = history or History()
        self.tools = Tool.resolve(tools)
        self.formatting = Formatting.resolve(formatting)
        self._state: dict[str, Any] = {}
    
    def __bool__(self) -> bool:
        return bool(self.history)
    
    def __len__(self) -> int:
        return len(self.history)
    
    def __iter__(self) -> Iterator[Interaction]:
        yield from self.history
    
    @overload
    def __getitem__(self, index: int) -> Interaction: ...
    
    @overload
    def __getitem__(self, index: slice) -> Self: ...
    
    def __getitem__(self, index: int | slice) -> Interaction | Self:
        if isinstance(index, slice):
            return self.replace(history=self.history[index])
        return self.history[index]
    
    def __aenter__(self) -> Self:
        return self
    
    async def __aexit__(
        self,
        exception: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.finish()
    
    @property
    def llm(self) -> LLM:
        return self.model.llm

    def replace(
        self,
        model: LLMModel | None = None,
        prompt: str | None = None,
        context: dict[str, Any] | None = None,
        history: HistoryArgument = None,
        tools: ToolsArgument = None,
        formatting: FormattingArgument = None,
    ) -> Self:
        return type(self)(
            model=model or self.model,
            prompt=prompt or self.prompt,
            context=context or self.context,
            history=history or self.history,
            tools=tools or self.tools,
            formatting=formatting or self.formatting,
        )

    def to_messages(self, format: bool = True) -> Messages:
        if format:
            prompt = self._format(self.prompt)
        else:
            prompt = self.prompt
        messages = self.history.to_messages()
        messages.insert(0, SystemMessage(content=prompt))
        return messages
    
    async def finish(self) -> None:
        await self.llm.teardown(self._state)
    
    async def complete(
        self,
        message: str | None = None,
        context: dict[str, Any] | None = None,
        *,
        save: bool = True,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        formatting: FormattingArgument = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
    ) -> str:
        completion = Completion(
            model=self.model,
            system_prompt=self.prompt,
            user_message=message,
            context=self.context | (context or {}),
            formatting=formatting or self.formatting,
            history=self.history,
            files=files,
            tools=self.tools | Tool.resolve(tools),
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
        )
        async with self._interaction(completion, save=save):
            await self.llm.complete(completion)
        return completion.text
    
    async def stream(
        self,
        message: str | None = None,
        context: dict[str, Any] | None = None,
        *,
        save: bool = True,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        formatting: FormattingArgument = None,
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
        stream = Stream(
            model=self.model,
            system_prompt=self.prompt,
            user_message=message,
            context=self.context | (context or {}),
            formatting=formatting or self.formatting,
            history=self.history,
            files=files,
            tools=self.tools | Tool.resolve(tools),
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
        )
        async with self._interaction(stream, save=save):
            await self.llm.stream(stream)
            async for chunk in stream:
                yield chunk
    
    @overload
    async def construct(
        self,
        schema: dict[str, Any],
        message: str | None = None,
        context: dict[str, Any] | None = None,
        *,
        save: bool = True,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        formatting: FormattingArgument = None,
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
        message: str | None = None,
        context: dict[str, Any] | None = None,
        *,
        save: bool = True,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        formatting: FormattingArgument = None,
        temperature: float | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        prediction: T | None = None,
    ) -> T: ...

    async def construct[T: BaseModel](
        self,
        schema: type[T] | dict[str, Any],
        message: str | None = None,
        context: dict[str, Any] | None = None,
        *,
        save: bool = True,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        formatting: FormattingArgument = None,
        temperature: float | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        prediction: T | dict[str, Any] | None = None,
    ) -> T | dict[str, Any]:
        if isinstance(schema, dict):
            model_class = cast(type[T], schema_to_model(schema))
            return_model = False
        else:
            model_class = schema
            return_model = True
        structured_output = StructuredOutput(
            model=self.model,
            schema=model_class,
            system_prompt=self.prompt,
            user_message=message,
            context=self.context | (context or {}),
            formatting=formatting or self.formatting,
            history=self.history,
            files=files,
            tools=self.tools | Tool.resolve(tools),
            use_tool=use_tool,
            temperature=temperature,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            top_k=top_k,
            prediction=prediction,
        )
        async with self._interaction(structured_output, save=save):
            await self.llm.construct(structured_output)
        if return_model:
            return structured_output.object
        return structured_output.dict

    async def classify(
        self,
        question: str,
        answers: list[str],
        message: str | None = None,
        context: dict[str, Any] | None = None,
        *,
        save: bool = False,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        formatting: FormattingArgument = None,
    ) -> str:
        classification = Classification(
            model=self.model,
            question=question,
            answers=answers,
            user_message=message,
            context=self.context | (context or {}),
            formatting=formatting or self.formatting,
            history=self.history,
            files=files,
            tools=self.tools | Tool.resolve(tools),
            use_tool=use_tool,
        )
        async with self._interaction(classification, save=save):
            await self.llm.classify(classification)
        return classification.answer
    
    @asynccontextmanager
    async def _interaction(self, operation: LLMOperation, save: bool) -> AsyncIterator[None]:
        interaction = Interaction(operation.user_content, operation.files)
        await self.llm.setup(operation, self._state)
        yield
        if not save:
            return
        ttft: float | None = None
        match operation:
            case Stream():
                assistant = operation.text
                ttft = operation.ttft
            case StructuredOutput():
                assistant = json.dumps(operation.dict)
            case Classification():
                assistant = operation.answer
            case Completion():
                assistant = operation.text
        interaction.end(assistant, operation.calls, ttft)
        self.history.append(interaction)