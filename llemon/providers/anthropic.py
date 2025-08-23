from __future__ import annotations

import json
import logging
from typing import AsyncIterator, ClassVar, Literal, NoReturn, cast

import anthropic
from anthropic.types import (
    Base64ImageSourceParam,
    Base64PDFSourceParam,
    ContentBlockParam,
    DocumentBlockParam,
    ImageBlockParam,
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
)
from pydantic import BaseModel

from llemon.apis.llm.llm import LLM
from llemon.apis.llm.llm_model import LLMModel
from llemon.apis.llm.llm_model_property import LLMModelProperty
from llemon.apis.llm.llm_tokenizer import LLMTokenizer
from llemon.errors import ConfigurationError, Error
from llemon.models.file import File
from llemon.models.generate import GenerateRequest, GenerateResponse
from llemon.models.generate_object import GenerateObjectRequest, GenerateObjectResponse
from llemon.models.generate_stream import GenerateStreamRequest, GenerateStreamResponse
from llemon.models.tool import Call, Tool
from llemon.types import NS, ToolCalls, ToolStream
from llemon.utils.logs import ASSISTANT, SYSTEM, USER

STRUCTURED_OUTPUT = "structured_output"
log = logging.getLogger(__name__)


class Anthropic(LLM):

    default_max_tokens: ClassVar[int] = 4096

    opus41 = LLMModelProperty("claude-opus-4-1")
    opus4 = LLMModelProperty("claude-opus-4-0")
    sonnet4 = LLMModelProperty("claude-sonnet-4-0")
    sonnet37 = LLMModelProperty("claude-3-7-sonnet-latest")
    sonnet35 = LLMModelProperty("claude-3-5-sonnet-latest")
    haiku35 = LLMModelProperty("claude-3-5-haiku-latest")
    haiku3 = LLMModelProperty("claude-3-haiku-20240307")

    def __init__(self, api_key: str) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    def get_tokenizer(self, model: LLMModel) -> LLMTokenizer:
        return AnthropicTokenizer(self.client, model)

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return await self._generate(request, GenerateResponse(request))

    async def generate_stream(self, request: GenerateStreamRequest) -> GenerateStreamResponse:
        return await self._generate_stream(request, GenerateStreamResponse(request))

    async def generate_object[T: BaseModel](self, request: GenerateObjectRequest[T]) -> GenerateObjectResponse[T]:
        return await self._generate_object(request, GenerateObjectResponse(request))

    async def _messages(self, request: GenerateRequest) -> list[MessageParam]:
        messages: list[MessageParam] = []
        if request.instructions:
            log.debug(SYSTEM + "%s", await request.render_instructions())
        if request.history:
            self._log_history(request.history)
            for request_, response_ in request.history:
                if not isinstance(request_, GenerateRequest) or not isinstance(response_, GenerateResponse):
                    continue
                messages.append(await self._user(request_.user_input, request_.files))
                if response_.calls:
                    messages.append(self._tool_call(response_.calls))
                    messages.append(self._tool_results(response_.calls))
                messages.append(self._assistant(response_.text))
        log.debug(USER + "%s", request.user_input)
        messages.append(await self._user(request.user_input, request.files))
        return messages

    async def _user(self, text: str, files: list[File]) -> MessageParam:
        content: str | list[ContentBlockParam]
        if files:
            content = []
            if text:
                content.append(TextBlockParam(type="text", text=text))
            for file in files:
                if file.is_image:
                    content.append(self._image(file))
                else:
                    content.append(self._document(file))
        else:
            content = text
        return MessageParam(role="user", content=content)

    def _image(self, file: File) -> ImageBlockParam:
        if file.data:
            return ImageBlockParam(
                type="image",
                source=Base64ImageSourceParam(
                    type="base64",
                    data=file.base64,
                    media_type=file.mimetype,  # type: ignore
                ),
            )
        return ImageBlockParam(
            type="image",
            source=URLImageSourceParam(
                type="url",
                url=file.url,
            ),
        )

    def _document(self, file: File) -> DocumentBlockParam:
        if file.data:
            return DocumentBlockParam(
                type="document",
                source=Base64PDFSourceParam(
                    type="base64",
                    data=file.base64,
                    media_type="application/pdf",
                ),
            )
        return DocumentBlockParam(
            type="document",
            source=URLPDFSourceParam(
                type="url",
                url=file.url,
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
            instructions = await request.render_instructions() if request.instructions else anthropic.NOT_GIVEN
            anthropic_response = await self.client.messages.create(
                model=request.model.name,
                messages=messages,
                max_tokens=request.max_tokens or request.model.config.max_output_tokens or self.default_max_tokens,
                system=instructions,
                temperature=_optional(request.temperature),
                top_p=_optional(request.top_p),
                top_k=_optional(request.top_k),
                stop_sequences=_optional(request.stop),
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
            )
        except anthropic.APIError as error:
            raise Error(error)
        self._check_stop_reason(anthropic_response.stop_reason, request.return_incomplete_message)
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
            raise ValueError(f"no text in response from {self}")
        log.debug(ASSISTANT + "%s", text)
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
            instructions = await request.render_instructions() if request.instructions else anthropic.NOT_GIVEN
            result = self.client.messages.stream(
                model=request.model.name,
                messages=messages,
                max_tokens=request.max_tokens or request.model.config.max_output_tokens or self.default_max_tokens,
                system=instructions,
                temperature=_optional(request.temperature),
                top_p=_optional(request.top_p),
                top_k=_optional(request.top_k),
                stop_sequences=_optional(request.stop),
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
            )
        except anthropic.APIError as error:
            raise Error(error)

        async def stream() -> AsyncIterator[str]:
            tool_stream: ToolStream = {}
            async with result as events:
                async for event in events:
                    if event.type == "message_delta":
                        self._check_stop_reason(event.delta.stop_reason, request.return_incomplete_message)
                        if event.delta.stop_reason:
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
                log.debug(ASSISTANT + "%s", response.text)

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
        tool = Tool(
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
            instructions = await request.render_instructions() if request.instructions else anthropic.NOT_GIVEN
            anthropic_response = await self.client.messages.create(
                model=request.model.name,
                messages=messages,
                max_tokens=request.max_tokens or request.model.config.max_output_tokens or self.default_max_tokens,
                system=instructions,
                temperature=_optional(request.temperature),
                top_p=_optional(request.top_p),
                top_k=_optional(request.top_k),
                stop_sequences=_optional(request.stop),
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
            )
        except anthropic.APIError as error:
            raise Error(error)
        self._check_stop_reason(anthropic_response.stop_reason, return_incomplete_messages=False)
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
            raise Error(f"no object in response from {self}")
        log.debug(ASSISTANT + "%s", object)
        response.complete_object(request.schema.model_validate(object))
        return response

    def _tools(self, request: GenerateRequest) -> list[ToolParam] | anthropic.NotGiven:
        if not request.tools:
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
        reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"] | None,
        return_incomplete_messages: bool,
    ) -> None:
        if reason == "refusal":
            raise Error(f"response from {self} was blocked")
        if reason == "max_tokens" and not return_incomplete_messages:
            raise Error(f"response from {self} exceeded the maximum length")
        if reason == "pause_turn" and not return_incomplete_messages:
            raise Error(f"response from {self} took too long")

    async def _run_tools(
        self,
        request: GenerateRequest,
        response: GenerateResponse,
        messages: list[MessageParam],
        tool_calls: ToolCalls,
    ) -> None:
        calls = [Call(id, request.tools_dict[name], args) for id, name, args in tool_calls]
        await Call.async_run_all(calls)
        messages.append(self._tool_call(calls))
        messages.append(self._tool_results(calls))
        response.calls.extend(calls)


class AnthropicTokenizer(LLMTokenizer):

    def __init__(self, client: anthropic.AsyncAnthropic, model: LLMModel) -> None:
        self.client = client
        self.model = model
    
    async def count(self, text: str) -> int:
        response = await self.client.messages.count_tokens(
            model=self.model.name,
            messages=[MessageParam(role="user", content=text)],
        )
        return response.input_tokens
    
    async def parse(self, text: str) -> NoReturn:
        raise self._unsupported()

    async def encode(self, *texts: str) -> NoReturn:
        raise self._unsupported()
    
    async def decode(self, ids: list[int]) -> NoReturn:
        raise self._unsupported()
    
    def _unsupported(self) -> ConfigurationError:
        raise ConfigurationError("Anthropic does not support explicit tokenization")


def _optional[T](value: T | None) -> T | anthropic.NotGiven:
    return value if value is not None else anthropic.NOT_GIVEN
