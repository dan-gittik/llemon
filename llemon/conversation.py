from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncGenerator, ClassVar, Iterator, Literal, Self, overload

import jinja2
from pydantic import BaseModel

from .interaction import History, Interaction
from .requests import CompletionRequest, ClassificationRequest
from .schema import schema_to_model
from .tool import Tool
from .types import FormattingArgument, FilesArgument, Messages, ToolsArgument
from .utils import now

if TYPE_CHECKING:
    from .llm import LLM
    from .model import Model

# TODO easily extend env


class Conversation:

    _env: jinja2.Environment | None

    def __init__(
        self,
        model: Model,
        prompt: str,
        context: dict[str, Any] | None = None,
        tools: ToolsArgument = None,
        history: History | None = None,
        formatting: FormattingArgument = True,
    ) -> None:
        if context is None:
            context = {}
        self.model = model
        self.prompt = prompt
        self.context = context
        self.tools = Tool.resolve(tools)
        self.history = history or History()
        self.formatting = Formatting.resolve(formatting)
        if self.formatting:
            self._env = jinja2.Environment(
                variable_start_string=self.formatting.variable_start,
                variable_end_string=self.formatting.variable_end,
                block_start_string=self.formatting.block_start,
                block_end_string=self.formatting.block_end,
                comment_start_string=self.formatting.comment_start,
                comment_end_string=self.formatting.comment_end,
            )
        else:
            self._env = None
    
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
    
    @property
    def llm(self) -> LLM:
        return self.model.llm

    def replace(
        self,
        model: Model | None = None,
        prompt: str | None = None,
        context: dict[str, Any] | None = None,
        history: History | None = None,
        tools: ToolsArgument | None = None,
        formatting: FormattingArgument | None = None,
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
        messages.insert(0, {"role": "system", "content": prompt})
        return messages
    
    async def complete(
        self,
        message: str | None = None,
        context: dict[str, Any] | None = None,
        *,
        save: bool = True,
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
    ) -> str:
        request = CompletionRequest(
            model=self.model.name,
            system_prompt=self._format(self.prompt, context),
            user_message=message,
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
        interaction = Interaction(request.user_content, request.files)
        response = await self.llm.complete(request)
        content = response.contents[0]
        if save:
            interaction.end(content, response.calls)
            self.history.append(interaction)
        return content
    
    async def stream(
        self,
        message: str | None = None,
        context: dict[str, Any] | None = None,
        *,
        save: bool = True,
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
    ) -> AsyncGenerator[str, None]:
        request = CompletionRequest(
            model=self.model.name,
            system_prompt=self._format(self.prompt, context),
            user_message=message,
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
        interaction = Interaction(request.user_content, request.files)
        ttft: float | None = None
        response = await self.llm.stream(request)
        async for chunk in response:
            if ttft is None:
                ttft = (now() - interaction.started).total_seconds()
            yield chunk
        if save:
            interaction.end(response.output, response.calls, ttft)
            self.history.append(interaction)
    
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
        temperature: float | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        prediction: T | dict[str, Any] | None = None,
    ) -> T | dict[str, Any]:
        if isinstance(schema, dict):
            model_class = schema_to_model(schema)
        else:
            model_class = schema
        request = CompletionRequest(
            model=self.model.name,
            system_prompt=self._format(self.prompt, context),
            user_message=message,
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
        interaction = Interaction(request.user_content, request.files)
        response = await self.llm.construct(model_class, request)
        output = response.outputs[0]
        if save:
            interaction.end(output.model_dump_json(), response.calls)
            self.history.append(interaction)
        if isinstance(schema, dict):
            return output.model_dump()
        return output  # type: ignore

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
    ) -> str:
        request = ClassificationRequest(
            model=self.model.name,
            question=question,
            answers=answers,
            user_message=message,
            history=self.history,
            files=files,
            tools=self.tools | Tool.resolve(tools),
            use_tool=use_tool,
        )
        interaction = Interaction(request.user_content, request.files)
        response = await self.llm.classify(request)
        if save:
            interaction.end(response.answer, response.calls)
            self.history.append(interaction)
        return response.answer
    
    def _format(self, text: str, context: dict[str, Any] | None = None) -> str:
        if not self._env:
            return text
        template = self._env.from_string(text)
        return template.render(self.context | (context or {}))
    

class Formatting(BaseModel):
    variable_start: str = "{{"
    variable_end: str = "}}"
    block_start: str = "{%"
    block_end: str = "%}"
    comment_start: str = "{#"
    comment_end: str = "#}"

    brackets: ClassVar[dict[str, str]] = {
        "(": ")",
        "[": "]",
        "{": "}",
        "<": ">",
    }

    @classmethod
    def resolve(cls, formatting: FormattingArgument) -> Formatting | Literal[False]:
        if formatting is False:
            return False
        if formatting is True:
            return cls()
        if isinstance(formatting, str):
            return cls.from_bracket(formatting)
        return formatting

    @classmethod
    def from_bracket(self, start: str) -> Formatting:
        if start not in self.brackets:
            raise ValueError(f"Invalid bracket {start!r} (expected '(', '{{', '[' or '<')")
        end = self.brackets[start]
        return self(
            variable_start=start * 2,
            variable_end=end * 2,
            block_start=f"{start}%",
            block_end=f"%{end}",
            comment_start=f"{start}#",
            comment_end=f"#{end}",
        )