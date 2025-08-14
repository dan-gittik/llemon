from __future__ import annotations

import json
import logging
from typing import Any, ClassVar, Literal, AsyncIterator

import anthropic
from anthropic.types import (
    MessageParam,
    ImageBlockParam,
    DocumentBlockParam,
    ToolParam,
    ToolChoiceToolParam,
)
from pydantic import BaseModel

from ..file import File
from ..llm import LLM, LLMModelGetter
from ..protocol import Completion, Stream, StructuredOutput, LLMOperation
from ..tool import Call, ToolRecord
from ..utils import USER, ASSISTANT, Error

log = logging.getLogger(__name__)


class Anthropic(LLM):

    default_max_tokens: ClassVar[int] = 4096

    opus41 = LLMModelGetter("claude-opus-4-1")
    opus4 = LLMModelGetter("claude-opus-4-0")
    sonnet4 = LLMModelGetter("claude-sonnet-4-0")
    sonnet37 = LLMModelGetter("claude-3-7-sonnet-latest")
    sonnet35 = LLMModelGetter("claude-3-5-sonnet-latest")
    haiku35 = LLMModelGetter("claude-3-5-haiku-latest")
    haiku3 = LLMModelGetter("claude-3-haiku-20240307")

    def __init__(self, api_key: str) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def complete(self, completion: Completion) -> None:
        await self._complete(completion, await self._messages(completion))
    
    async def stream(self, stream: Stream) -> None:
        await self._stream(stream, await self._messages(stream))
    
    async def construct[T: BaseModel](self, structured_output: StructuredOutput[T]) -> None:
        await self._construct(structured_output, await self._messages(structured_output))
    
    async def _messages(self, completion: Completion) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if completion.history:
            completion.history.log()
            for interaction in completion.history.interactions:
                messages.append(await self._user(interaction.user, interaction.files))
                if interaction.calls:
                    messages.append(self._tool_call(interaction.calls))
                    messages.append(self._tool_results(interaction.calls))
                messages.append(self._assistant(interaction.assistant))
        user_message = completion.get_user_message()
        log.debug(USER + "%s", user_message)
        messages.append(await self._user(user_message, completion.files))
        return messages
    
    async def _user(self, text: str, files: list[File]) -> MessageParam:
        content: str | list[dict[str, Any]]
        if files:
            content = []
            if text:
                content.append({"type": "text", "text": text})
            for file in files:
                if file.is_image:
                    if file.data:
                        content.append(ImageBlockParam(
                            type="image",
                            source={
                                "type": "base64",
                                "media_type": file.mimetype,
                                "data": file.base64, 
                            },
                        ))
                    else:
                        content.append(DocumentBlockParam(
                            type="image",
                            source={"type": "url", "url": file.url},
                        ))
                else:
                    if file.data:
                        content.append(DocumentBlockParam(
                            type="document",
                            source={
                                "type": "base64",
                                "media_type": file.mimetype,
                                "data": file.base64,
                            },
                        ))
                    else:
                        content.append(DocumentBlockParam(
                            type="document",
                            source={"type": "url", "url": file.url},
                        ))
        else:
            content = text
        return MessageParam(role="user", content=content)
    
    def _assistant(self, content: str) -> MessageParam:
        return MessageParam(role="assistant", content=content)
    
    def _tool_call(self, calls: list[Call]) -> MessageParam:
        return MessageParam(
            role="assistant",
            content=[
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.tool.name,
                    "input": call.arguments,
                }
                for call in calls
            ]
        )

    def _tool_results(self, calls: list[Call]) -> MessageParam:
        return MessageParam(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": call.result_json,
                }
                for call in calls
            ]
        )

    async def _complete(self, completion: Completion, messages: list[MessageParam]) -> None:
        try:
            response = await self.client.messages.create(
                model=completion.model.name,
                messages=messages,
                max_tokens=completion.max_tokens or self.default_max_tokens,
                system=_optional(completion.system_prompt),
                temperature=_optional(completion.temperature),
                top_p=_optional(completion.top_p),
                top_k=_optional(completion.top_k),
                stop_sequences=_optional(completion.stop),
                tools=self._tools(completion),
                tool_choice=self._tool_choice(completion),
            )
        except anthropic.APIError as error:
            raise Error(error)
        self._check_stop_reason(response.stop_reason)
        texts: list[str] = []
        tools: list[tuple[str, str, dict[str, Any]]] = []
        for content in response.content:
            if content.type == "tool_use":
                tools.append((content.id, content.name, content.input))
            if content.type == "text":
                texts.append(content.text)
        if tools:
            await self._run_tools(completion, messages, tools)
            return await self._complete(completion, messages)
        text = "".join(texts)
        if not text:
            raise ValueError(f"no text in response from {self}")
        log.debug(ASSISTANT + "%s", text)
        completion.add_text(text)
    
    async def _stream(self, stream: Stream, messages: list[MessageParam]) -> None:
        try:
            response = self.client.messages.stream(
                model=stream.model.name,
                messages=messages,
                max_tokens=stream.max_tokens or self.default_max_tokens,
                system=_optional(stream.system_prompt),
                temperature=_optional(stream.temperature),
                top_p=_optional(stream.top_p),
                top_k=_optional(stream.top_k),
                stop_sequences=_optional(stream.stop),
                tools=self._tools(stream),
                tool_choice=self._tool_choice(stream),
            )
        except anthropic.APIError as error:
            raise Error(error)

        async def chunks() -> AsyncIterator[str]:
            tool_stream: dict[int, tuple[str, str, list[str]]] = {}
            async with response as events:
                async for event in events:
                    if event.type == "message_delta":
                        self._check_stop_reason(event.delta.stop_reason)
                        if event.delta.stop_reason:
                            break
                    if event.type == "message_stop":
                        break
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            tool_stream[event.index] = block.id, block.name, []
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta" and delta.text:
                            yield delta.text
                        if delta.type == "input_json_delta":
                            tool_stream[event.index][2].append(delta.partial_json)
                    if event.type == "content_block_stop":
                        block = event.content_block
            if tool_stream:
                yield "\n"
                tools = [(id, name, json.loads("".join(args))) for (id, name, args) in tool_stream.values()]
                await self._run_tools(stream, messages, tools)
                await self._stream(stream, messages)
                async for chunk in stream.stream:
                    yield chunk
            else:
                log.debug(ASSISTANT + "%s", stream.text)
        
        stream.stream = chunks()
    
    async def _construct[T: BaseModel](
        self,
        structured_output: StructuredOutput[T],
        messages: list[MessageParam],
    ) -> None:
        structured_output.tools["structured_output"] = ToolRecord({
            "name": "structured_output",
            "description": "Use this tool to output a structured object",
            "parameters": structured_output.schema.model_json_schema(),
        })
        structured_output.append_instruction("""
            Use the structured_output tool to output a structured object.
        """)
        try:
            response = await self.client.messages.create(
                model=structured_output.model.name,
                messages=messages,
                max_tokens=structured_output.max_tokens or self.default_max_tokens,
                system=_optional(structured_output.system_prompt),
                temperature=_optional(structured_output.temperature),
                top_p=_optional(structured_output.top_p),
                top_k=_optional(structured_output.top_k),
                stop_sequences=_optional(structured_output.stop),
                tools=self._tools(structured_output),
                tool_choice=self._tool_choice(structured_output),
            )
        except anthropic.APIError as error:
            raise Error(error)
        self._check_stop_reason(response.stop_reason)
        object_: dict[str, Any] = []
        tools: list[tuple[str, str, dict[str, Any]]] = []
        for content in response.content:
            if content.type == "tool_use":
                if content.name == "structured_output":
                    object_ = content.input
                else:
                    tools.append((content.id, content.name, content.input))
        if tools:
            await self._run_tools(structured_output, messages, tools)
            return await self._construct(structured_output, messages)
        if not object_:
            raise Error(f"no objects in response from {self}")
        log.debug(ASSISTANT + "%s", object_)
        structured_output.add_object(object_)

    def _tools(self, operation: LLMOperation) -> list[ToolParam] | anthropic.NotGiven:
        if not operation.tools:
            return anthropic.NOT_GIVEN
        tools: list[ToolParam] = []
        for tool in operation.tools.values():
            tools.append({
                "name": tool.schema["name"],
                "description": tool.schema["description"],
                "input_schema": tool.schema["parameters"],
            })
        return tools
    
    def _tool_choice(
        self,
        operation: LLMOperation,
    ) -> anthropic.NotGiven | Literal["none"] | Literal["any"] | ToolChoiceToolParam:
        if operation.use_tool is None:
            return anthropic.NOT_GIVEN
        if operation.use_tool is False:
            return ToolChoiceToolParam(type="none")
        if operation.use_tool is True:
            return ToolChoiceToolParam(type="any")
        return ToolChoiceToolParam(type="tool", name=operation.use_tool, disable_parallel_tool_use=True)
    
    def _check_stop_reason(self, reason: str) -> None:
        if reason == "refusal":
            raise Error(f"response from {self} was refused")
        if reason == "max_tokens" and not self.return_incomplete_messages:
            raise Error(f"response from {self} exceeded the maximum length")
    
    async def _run_tools(
        self,
        operation: LLMOperation,
        messages: list[dict[str, str]],
        tool_calls: list[tuple[str, str, dict[str, Any]]],
    ) -> None:
        calls = [Call(id, operation.tools[name], args) for id, name, args in tool_calls]
        await Call.async_run_all(calls)
        messages.append(self._tool_call(calls))
        messages.append(self._tool_results(calls))
        operation.calls.extend(calls)
    

def _optional[T](value: T | None) -> T | anthropic.NotGiven:
    return value if value is not None else anthropic.NOT_GIVEN