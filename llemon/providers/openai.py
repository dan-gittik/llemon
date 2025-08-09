from __future__ import annotations

from typing import Any, ClassVar, Iterator, Literal, cast, overload

import openai
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolParam,
    ParsedChatCompletion,
    ParsedChoice,
)
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

from ..file import File
from ..llm import LLM
from ..model import ModelGetter
from ..requests import Call, CompletionRequest, CompletionResponse, Request, Response, StructuredOutputResponse
from ..utils import Error


class OpenAI(LLM):

    no_content: ClassVar[str] = "."

    # models
    gpt_4o = ModelGetter("gpt-4o")
    # /models

    def __init__(self, api_key: str) -> None:
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return await self._complete(
            request=request,
            response=CompletionResponse(),
            messages=self._messages(request),
        )
    
    async def construct[T: BaseModel](self, schema: type[T], request: CompletionRequest) -> StructuredOutputResponse[T]:
        return await self._construct(
            schema=schema,
            request=request,
            response=StructuredOutputResponse[T](),
            messages=self._messages(request),
        )

    def _messages(self, request: CompletionRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append(self._system(request.system_prompt))
        if request.history:
            for interaction in request.history.interactions:
                messages.append(self._user(interaction.user, interaction.files))
                if interaction.calls:
                    messages.append(self._tool_call(interaction.calls))
                    messages.extend(self._tool_results(interaction.calls))
                messages.append(self._assistant(interaction.assistant))
        messages.append(self._user(request.user_message, request.files))
        return messages
    
    def _system(self, prompt: str) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(role="system", content=prompt)
    
    def _user(self, text: str | None, files: list[File]) -> ChatCompletionUserMessageParam:
        images: list[str] = []
        for file in files:
            if not file.mimetype.startswith("image/"):
                raise ValueError(f"{self} does not support non-image files like {file}")
            images.append(file.url)
        content: str | list[dict[str, Any]]
        if images:
            content = []
            if text:
                content.append({"type": "text", "text": text})
            for image in images:
                content.append({"type": "image_url", "image_url": {"url": image}})
        else:
            content = text or CompletionRequest.no_content
        return ChatCompletionUserMessageParam(role="user", content=content)
    
    def _assistant(self, content: str) -> ChatCompletionAssistantMessageParam:
        return ChatCompletionAssistantMessageParam(role="assistant", content=content)
    
    def _tool_call(self, calls: list[Call]) -> ChatCompletionAssistantMessageParam:
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            tool_calls=[
                {
                    "type": "function",
                    "id": call.id,
                    "function": {"name": call.tool.name, "arguments": call.arguments_json},
                }
                for call in calls
            ]
        )

    def _tool_results(self, calls: list[Call]) -> list[ChatCompletionToolMessageParam]:
        return [
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=call.id,
                content=call.return_value_string,
            )
            for call in calls
        ]
    
    async def _complete(
        self,
        request: CompletionRequest,
        response: CompletionResponse,
        messages: list[ChatCompletionMessageParam],
    ) -> CompletionResponse:
        try:
            completion = await self.client.chat.completions.create(
                model=request.model,
                messages=messages,
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
                temperature=_optional(request.temperature),
                max_tokens=_optional(request.max_tokens),
                seed=_optional(request.seed),
                frequency_penalty=_optional(request.frequency_penalty),
                presence_penalty=_optional(request.presence_penalty),
                n=_optional(request.n),
                top_p=_optional(request.top_p),
                stop=_optional(request.stop),
            )
        except openai.APIError as error:
            raise Error(error)
        for choice in self._choices(completion):
            if choice.finish_reason == "tool_calls":
                await self._run_tools(request, response, messages, choice.message)
                return await self._complete(request, response, messages)
            if not choice.message.content:
                raise Error(f"no content in response from {self}")
            response.contents.append(cast(str, choice.message.content))
        return response

    async def _construct[T: BaseModel](
        self,
        schema: type[T],
        request: CompletionRequest,
        response: StructuredOutputResponse[T],
        messages: list[ChatCompletionMessageParam],
    ) -> StructuredOutputResponse[T]:
        try:
            construction = await self.client.beta.chat.completions.parse(
                model=request.model,
                response_format=schema,
                messages=messages,
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
                temperature=_optional(request.temperature),
                max_tokens=_optional(request.max_tokens),
                seed=_optional(request.seed),
                frequency_penalty=_optional(request.frequency_penalty),
                presence_penalty=_optional(request.presence_penalty),
                n=_optional(request.n),
                top_p=_optional(request.top_p),
                stop=_optional(request.stop),
            )
        except openai.APIError as error:
            raise Error(error)
        for choice in self._choices(construction):
            if choice.finish_reason == "tool_calls":
                await self._run_tools(request, response, messages, choice.message)
                return await self._construct(schema, request, response, messages)
            response.outputs.append(cast(T, choice.message.parsed))
        return response
    
    def _tools(self, request: CompletionRequest) -> list[ChatCompletionToolParam] | openai.NotGiven:
        if not request.tools:
            return openai.NOT_GIVEN
        tools: list[ChatCompletionToolParam] = []
        for tool in request.tools.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.schema["name"],
                    "description": tool.schema["description"],
                    "parameters": tool.schema["parameters"],
                    "strict": True,
                }
            })
        return tools
    
    def _tool_choice(
        self,
        request: CompletionRequest,
    ) -> openai.NotGiven | Literal["none"] | Literal["required"] | ChatCompletionNamedToolChoiceParam:
        if request.use_tool is None:
            return openai.NOT_GIVEN
        if request.use_tool is False:
            return "none"
        if request.use_tool is True:
            return "required"
        return {"type": "function", "function": {"name": request.use_tool}}

    @overload
    def _choices(self, response: ParsedChatCompletion) -> Iterator[ParsedChoice]: ...

    @overload
    def _choices(self, response: ChatCompletion) -> Iterator[Choice]: ...

    def _choices(self, response: ChatCompletion | ParsedChatCompletion) -> Iterator[Choice | ParsedChoice]:
        if not response.choices:
            raise Error(f"no response from {self}")
        for choice in response.choices:
            if choice.finish_reason == "content_filter":
                raise Error(f"response from {self} was filtered")
            if choice.finish_reason == "length":
                raise Error(f"response from {self} exceeded the maximum length")
            if choice.message.refusal:
                raise Error(f"response from {self} was refused: {choice.message.refusal}")
            yield choice

    async def _run_tools(
        self,
        request: Request,
        response: Response,
        messages: list[dict[str, str]],
        message: ChatCompletionMessage,
    ) -> None:
        calls: list[Call] = []
        for tool_call in message.tool_calls or []:
            call = Call(tool_call.id, request.tools[tool_call.function.name], tool_call.function.arguments)
            calls.append(call)
        await Call.async_run_all(calls)
        messages.append(self._tool_call(calls))
        messages.extend(self._tool_results(calls))
        response.calls.extend(calls)


def _optional[T](value: T | None) -> T | openai.NotGiven:
    return value if value is not None else openai.NOT_GIVEN