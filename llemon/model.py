from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, cast, overload

from pydantic import BaseModel

from .conversation import Conversation
from .requests import ClassificationRequest, CompletionRequest
from .schema import schema_to_model
from .types import FilesArgument, HistoryArgument, ToolsArgument, FormattingArgument

if TYPE_CHECKING:
    from .llm import LLM


class ModelGetter:

    def __init__(self, name: str) -> None:
        self.name = name
    
    def __get__(self, instance: LLM, owner: type[LLM]) -> Model:
        return owner.get(self.name)


class Model:

    def __init__(self, llm: LLM, name: str) -> None:
        self.llm = llm
        self.name = name
    
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
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        n: None,
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
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        n: int,
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
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        n: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
    ) -> str | list[str]:
        system_prompt, user_message = self._resolve_messages(message1, message2)
        request = CompletionRequest(
            model=self.name,
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
            n=n,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            prediction=prediction,
        )
        response = await self.llm.complete(request)
        if n is None:
            return response.contents[0]
        return response.contents

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
        request = CompletionRequest(
            model=self.name,
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
        response = await self.llm.stream(request)
        async for chunk in response:
            yield chunk
    
    @overload
    async def construct(
        self,
        schema: dict[str, Any],
        message1: str | None = None,
        message2: str | None = None,
        /,
        *,
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        n: None,
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
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        n: None,
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
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        n: int,
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
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        n: int,
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
        history: HistoryArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        n: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        prediction: T | dict[str, Any] | None = None,
    ) -> T | dict[str, Any] | list[T] | list[dict[str, Any]]:
        if isinstance(schema, dict):
            model_class = schema_to_model(schema)
            return_model = False
        else:
            model_class = schema
            return_model = True
        system_prompt, user_message = self._resolve_messages(message1, message2)
        request = CompletionRequest(
            model=self.name,
            user_message=user_message,
            system_prompt=system_prompt,
            history=history,
            files=files,
            temperature=temperature,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            top_k=top_k,
            tools=tools,
            use_tool=use_tool,
            prediction=prediction,
        )
        response = await self.llm.construct(model_class, request)
        if n is None:
            if return_model:
                return cast(T, response.outputs[0])
            return response.outputs[0].model_dump()
        if return_model:
            return cast(list[T], response.outputs)
        return [output.model_dump() for output in response.outputs]

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
        request = ClassificationRequest(
            model=self.name,
            user_message=message,
            question=question,
            answers=answers,
            history=history,
            files=files,
            tools=tools,
            use_tool=use_tool,
        )
        response = await self.llm.classify(request)
        return response.answer

    def _resolve_messages(self, message1: str | None, message2: str | None) -> tuple[str | None, str | None]:
        if message2 is None:
            return None, message1
        return message1, message2