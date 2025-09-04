from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, AsyncIterator, ClassVar, Literal, cast

import anthropic
from anthropic.types import (
    Base64ImageSourceParam,
    Base64PDFSourceParam,
    CacheControlEphemeralParam,
    DocumentBlockParam,
    ImageBlockParam,
    MessageDeltaUsage,
    MessageParam,
    TextBlockParam,
    ToolChoiceAnyParam,
    ToolChoiceNoneParam,
    ToolChoiceToolParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    URLImageSourceParam,
    URLPDFSourceParam,
    Usage,
)
from pydantic import BaseModel

import llemon
from llemon.types import NS, ToolCalls, ToolStream
from llemon.utils import Emoji, async_parallelize

if TYPE_CHECKING:
    from llemon import (
        Call,
        File,
        GenerateObjectRequest,
        GenerateObjectResponse,
        GenerateRequest,
        GenerateResponse,
        GenerateStreamRequest,
        GenerateStreamResponse,
    )

log = logging.getLogger(__name__)


class Anthropic(llemon.LLMProvider):

    default_max_tokens: ClassVar[int] = 4096

    opus41 = llemon.LLMProperty("claude-opus-4-1")
    opus4 = llemon.LLMProperty("claude-opus-4-0")
    sonnet4 = llemon.LLMProperty("claude-sonnet-4-0")
    sonnet37 = llemon.LLMProperty("claude-3-7-sonnet-latest")
    sonnet35 = llemon.LLMProperty("claude-3-5-sonnet-latest")
    haiku35 = llemon.LLMProperty("claude-3-5-haiku-latest")
    haiku3 = llemon.LLMProperty("claude-3-haiku-20240307")

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def count_tokens(self, request: GenerateRequest) -> int:
        messages = await self._messages(request)
        response = await self.client.messages.count_tokens(
            model=request.llm.model,
            messages=messages,
        )
        return response.input_tokens

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return await self._generate(request, llemon.GenerateResponse(request))

    async def generate_stream(self, request: GenerateStreamRequest) -> GenerateStreamResponse:
        return await self._generate_stream(request, llemon.GenerateStreamResponse(request))

    async def generate_object[T: BaseModel](self, request: GenerateObjectRequest[T]) -> GenerateObjectResponse[T]:
        return await self._generate_object(request, llemon.GenerateObjectResponse(request))

    async def _messages(self, request: GenerateRequest) -> list[MessageParam]:
        messages: list[MessageParam] = []
        if request.instructions:
            log.debug(Emoji.SYSTEM + "%s", await request.render_instructions())
        if request.history:
            self._log_history(request.history)
            for request_, response_ in request.history:
                if not isinstance(request_, llemon.GenerateRequest):
                    continue
                assert isinstance(response_, llemon.GenerateResponse)
                messages.append(await self._user(request_.user_input, request_.files))
                if response_.calls:
                    messages.append(self._tool_call(response_.calls))
                    messages.append(self._tool_results(response_.calls))
                messages.append(self._assistant(response_.text))
        log.debug(Emoji.USER + "%s", request.user_input)
        messages.append(await self._user(request.user_input, request.files, cache=request.cache))
        return messages

    async def _system(self, request: GenerateRequest) -> anthropic.NotGiven | str | list[TextBlockParam]:
        if not request.instructions:
            return anthropic.NOT_GIVEN
        instructions = await request.render_instructions()
        if request.cache:
            return [
                TextBlockParam(
                    type="text",
                    text=instructions,
                    cache_control=CacheControlEphemeralParam(
                        type="ephemeral",
                    ),
                ),
            ]
        return instructions

    async def _user(self, text: str, files: list[File], cache: bool | None = None) -> MessageParam:
        content: list[TextBlockParam | ImageBlockParam | DocumentBlockParam] = []
        if files:
            if text:
                content.append(TextBlockParam(type="text", text=text))
            for file in files:
                if file.is_image:
                    content.append(self._image(file))
                else:
                    content.append(self._document(file))
        else:
            content.append(TextBlockParam(type="text", text=text))
        if cache:
            content[-1]["cache_control"] = CacheControlEphemeralParam(type="ephemeral")
        return MessageParam(role="user", content=content)

    def _image(self, file: File) -> ImageBlockParam:
        if file.is_url:
            return ImageBlockParam(
                type="image",
                source=URLImageSourceParam(
                    type="url",
                    url=file.url,
                ),
            )
        return ImageBlockParam(
            type="image",
            source=Base64ImageSourceParam(
                type="base64",
                data=file.base64,
                media_type=file.mimetype,  # type: ignore
            ),
        )

    def _document(self, file: File) -> DocumentBlockParam:
        if file.is_url:
            return DocumentBlockParam(
                type="document",
                source=URLPDFSourceParam(
                    type="url",
                    url=file.url,
                ),
            )
        return DocumentBlockParam(
            type="document",
            source=Base64PDFSourceParam(
                type="base64",
                data=file.base64,
                media_type="application/pdf",
            ),
        )

    def _assistant(self, content: str) -> MessageParam:
        return MessageParam(role="assistant", content=content)

    def _tool_call(self, calls: list[Call]) -> MessageParam:
        return MessageParam(
            role="assistant",
            content=[
                ToolUseBlockParam(
                    type="tool_use",
                    id=call.id,
                    name=call.tool.compatible_name,
                    input=call.arguments,
                )
                for call in calls
            ],
        )

    def _tool_results(self, calls: list[Call]) -> MessageParam:
        return MessageParam(
            role="user",
            content=[
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id=call.id,
                    content=call.result_json,
                )
                for call in calls
            ],
        )

    async def _generate(
        self,
        request: GenerateRequest,
        response: GenerateResponse,
        messages: list[MessageParam] | None = None,
    ) -> GenerateResponse:
        if messages is None:
            messages = await self._messages(request)
        try:
            anthropic_response = await self.client.messages.create(
                model=request.llm.model,
                messages=messages,
                max_tokens=request.max_tokens or request.llm.config.max_output_tokens or self.default_max_tokens,
                system=await self._system(request),
                temperature=_optional(request.temperature),
                top_p=_optional(request.top_p),
                top_k=_optional(request.top_k),
                stop_sequences=_optional(request.stop),
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
                timeout=_optional(request.timeout),
            )
        except anthropic.APIError as error:
            raise request.error(str(error))
        request.id = anthropic_response.id
        self._set_usage(response, anthropic_response.usage)
        self._check_stop_reason(request, anthropic_response.stop_reason)
        texts: list[str] = []
        tools: ToolCalls = []
        for content in anthropic_response.content:
            if content.type == "tool_use":
                tools.append((content.id, content.name, cast(NS, content.input)))
            if content.type == "text":
                texts.append(content.text)
        if tools:
            await self._run_tools(request, response, messages, tools)
            return await self._generate(request, response, messages=messages)
        text = "".join(texts)
        if not text:
            raise request.error(f"{request} response has no text")
        log.debug(Emoji.ASSISTANT + "%s", text)
        response.complete_text(text)
        return response

    async def _generate_stream(
        self,
        request: GenerateStreamRequest,
        response: GenerateStreamResponse,
        messages: list[MessageParam] | None = None,
    ) -> GenerateStreamResponse:
        if messages is None:
            messages = await self._messages(request)
        try:
            anthropic_response = self.client.messages.stream(
                model=request.llm.model,
                messages=messages,
                max_tokens=request.max_tokens or request.llm.config.max_output_tokens or self.default_max_tokens,
                system=await self._system(request),
                temperature=_optional(request.temperature),
                top_p=_optional(request.top_p),
                top_k=_optional(request.top_k),
                stop_sequences=_optional(request.stop),
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
                timeout=_optional(request.timeout),
            )
        except anthropic.APIError as error:
            raise request.error(str(error))

        async def stream() -> AsyncIterator[str]:
            tool_stream: ToolStream = {}
            async with anthropic_response as events:
                if events.request_id:
                    request.id = events.request_id
                async for event in events:
                    if event.type == "message_delta":
                        if not event.delta.stop_reason:
                            continue
                        self._set_usage(response, event.usage)
                        self._check_stop_reason(request, event.delta.stop_reason)
                        break
                    if event.type == "message_stop":
                        break
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            tool_stream[event.index] = block.id, block.name, []
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta" and event.delta.text:
                            yield event.delta.text
                        if event.delta.type == "input_json_delta":
                            tool_stream[event.index][2].append(event.delta.partial_json)
            if tool_stream:
                yield "\n"
                tools = [(id, name, json.loads("".join(args))) for (id, name, args) in tool_stream.values()]
                await self._run_tools(request, response, messages, tools)
                await self._generate_stream(request, response, messages=messages)
                assert response.stream is not None
                async for delta in response.stream:
                    yield delta
            else:
                response.complete_stream()
                log.debug(Emoji.ASSISTANT + "%s", response.text)

        response.stream = stream()
        return response

    async def _generate_object[T: BaseModel](
        self,
        request: GenerateObjectRequest[T],
        response: GenerateObjectResponse[T],
        messages: list[MessageParam] | None = None,
    ) -> GenerateObjectResponse[T]:
        if messages is None:
            messages = await self._messages(request)
        tool = llemon.Tool(
            name="structured_output",
            description="Use this tool to output a structured object",
            parameters=request.schema.model_json_schema(),
        )
        request.tools_dict[tool.compatible_name] = tool
        request.append_instruction(
            f"""
            Use the {tool.compatible_name} tool to output a structured object.
            """
        )
        try:
            anthropic_response = await self.client.messages.create(
                model=request.llm.model,
                messages=messages,
                max_tokens=request.max_tokens or request.llm.config.max_output_tokens or self.default_max_tokens,
                system=await self._system(request),
                temperature=_optional(request.temperature),
                top_p=_optional(request.top_p),
                top_k=_optional(request.top_k),
                stop_sequences=_optional(request.stop),
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
                timeout=_optional(request.timeout),
            )
        except anthropic.APIError as error:
            raise request.error(str(error))
        request.id = anthropic_response.id
        self._set_usage(response, anthropic_response.usage)
        self._check_stop_reason(request, anthropic_response.stop_reason)
        object: NS = {}
        tools: ToolCalls = []
        for content in anthropic_response.content:
            if content.type == "tool_use":
                if content.name == tool.compatible_name:
                    object = cast(NS, content.input)
                else:
                    tools.append((content.id, content.name, cast(NS, content.input)))
        if tools:
            await self._run_tools(request, response, messages, tools)
            return await self._generate_object(request, response, messages=messages)
        if not object:
            raise request.error(f"{request} response has no object")
        log.debug(Emoji.ASSISTANT + "%s", object)
        response.complete_object(request.schema.model_validate(object))
        return response

    def _tools(self, request: GenerateRequest) -> list[ToolParam] | anthropic.NotGiven:
        if not request.tools_dict:
            return anthropic.NOT_GIVEN
        tools: list[ToolParam] = []
        for tool in request.tools_dict.values():
            tools.append(
                ToolParam(
                    name=tool.compatible_name,
                    description=tool.description,
                    input_schema=tool.parameters,
                )
            )
        return tools

    def _tool_choice(
        self,
        request: GenerateRequest,
    ) -> anthropic.NotGiven | ToolChoiceNoneParam | ToolChoiceAnyParam | ToolChoiceToolParam:
        if request.use_tool is None:
            return anthropic.NOT_GIVEN
        if request.use_tool is False:
            return ToolChoiceNoneParam(type="none")
        if request.use_tool is True:
            return ToolChoiceAnyParam(type="any")
        return ToolChoiceToolParam(type="tool", name=request.get_tool_name(request.use_tool))

    def _check_stop_reason(
        self,
        request: GenerateRequest,
        reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"] | None,
    ) -> None:
        if reason == "refusal":
            raise request.error(f"{request} was blocked")
        if reason == "max_tokens" and not request.return_incomplete_message:
            raise request.error(f"{request} response exceeded the maximum length")
        if reason == "pause_turn" and not request.return_incomplete_message:
            raise request.error(f"{request} took too long")

    async def _run_tools(
        self,
        request: GenerateRequest,
        response: GenerateResponse,
        messages: list[MessageParam],
        tool_calls: ToolCalls,
    ) -> None:
        calls = [llemon.Call(id, request.tools_dict[name], args) for id, name, args in tool_calls]
        await async_parallelize(call.run for call in calls)
        messages.append(self._tool_call(calls))
        messages.append(self._tool_results(calls))
        response.calls.extend(calls)

    def _set_usage(self, response: GenerateResponse, usage: Usage | MessageDeltaUsage) -> None:
        response.input_tokens += usage.input_tokens or 0
        response.cache_tokens += usage.cache_creation_input_tokens or 0
        response.output_tokens += usage.output_tokens or 0


def _optional[T](value: T | None) -> T | anthropic.NotGiven:
    return value if value is not None else anthropic.NOT_GIVEN
