from __future__ import annotations

from typing import Any, ClassVar, Literal

import anthropic
from anthropic.types import (
    MessageParam,
    ImageBlockParam,
    DocumentBlockParam,
    ToolParam,
    ToolChoiceToolParam,
    ContentBlock,
    ToolUseBlock,
)
from pydantic import BaseModel

from ..file import File
from ..llm import LLM, LLMModelGetter
from ..protocol import Completion, StructuredOutput
from ..tool import Call, ToolRecord


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
    
    async def construct[T: BaseModel](self, structured_output: StructuredOutput[T]) -> None:
        await self._construct(structured_output, await self._messages(structured_output))
    
    async def _messages(self, completion: Completion) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if completion.history:
            for interaction in completion.history.interactions:
                messages.append(await self._user(interaction.user, interaction.files))
                if interaction.calls:
                    messages.append(self._tool_call(interaction.calls))
                    messages.append(self._tool_results(interaction.calls))
                messages.append(self._assistant(interaction.assistant))
        messages.append(await self._user(completion.get_user_message(), completion.files))
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
        text: list[str] = []
        tool_calls: list[ContentBlock] = []
        for content in response.content:
            if content.type == "tool_use":
                tool_calls.append(content)
            if content.type == "text":
                text.append(content.text)
        if tool_calls:
            await self._run_tools(completion, messages, tool_calls)
            return await self._complete(completion, messages)
        if not text:
            raise ValueError(f"no text in response from {self}")
        completion.add_text("\n".join(text))
    
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
        dicts: list[dict[str, Any]] = []
        tool_calls: list[ContentBlock] = []
        for content in response.content:
            if content.type == "tool_use":
                if content.name == "structured_output":
                    dicts.append(content.input)
                else:
                    tool_calls.append(content)
        if tool_calls:
            await self._run_tools(structured_output, messages, tool_calls)
            return await self._construct(structured_output, messages)
        if not dicts:
            raise ValueError(f"no models in response from {self}")
        structured_output.add_object(*dicts)

    def _tools(self, completion: Completion) -> list[ToolParam] | anthropic.NotGiven:
        if not completion.tools:
            return anthropic.NOT_GIVEN
        tools: list[ToolParam] = []
        for tool in completion.tools.values():
            tools.append({
                "name": tool.schema["name"],
                "description": tool.schema["description"],
                "input_schema": tool.schema["parameters"],
            })
        return tools
    
    def _tool_choice(
        self,
        completion: Completion,
    ) -> anthropic.NotGiven | Literal["none"] | Literal["any"] | ToolChoiceToolParam:
        if completion.use_tool is None:
            return anthropic.NOT_GIVEN
        if completion.use_tool is False:
            return ToolChoiceToolParam(type="none")
        if completion.use_tool is True:
            return ToolChoiceToolParam(type="any")
        return ToolChoiceToolParam(type="tool", name=completion.use_tool, disable_parallel_tool_use=True)

    async def _run_tools(
        self,
        completion: Completion,
        messages: list[dict[str, str]],
        tool_calls: list[ToolUseBlock],
    ) -> None:
        calls: list[Call] = []
        for tool_call in tool_calls:
            call = Call(tool_call.id, completion.tools[tool_call.name], tool_call.input)
            calls.append(call)
        await Call.async_run_all(calls)
        messages.append(self._tool_call(calls))
        messages.append(self._tool_results(calls))
        completion.calls.extend(calls)
    

def _optional[T](value: T | None) -> T | anthropic.NotGiven:
    return value if value is not None else anthropic.NOT_GIVEN