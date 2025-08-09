from __future__ import annotations

from typing import Any, ClassVar, Literal

import anthropic
from anthropic.types import MessageParam, ToolParam, ToolChoiceToolParam, ContentBlock, ToolUseBlock
from pydantic import BaseModel

from ..file import File
from ..llm import LLM
from ..model import ModelGetter
from ..requests import CompletionRequest, CompletionResponse, StructuredOutputResponse, Request, Response
from ..tool import Call, ToolSchema


class Anthropic(LLM):

    max_tokens: ClassVar[int] = 4096

    # models
    opus4 = ModelGetter("claude-opus-4-20250514")
    sonnet4 = ModelGetter("claude-sonnet-4-20250514")
    sonnet37 = ModelGetter("claude-3-7-sonnet-20250219")
    haiku35 = ModelGetter("claude-3-5-haiku-20241022")
    sonnet35v2 = ModelGetter("claude-3-5-sonnet-20241022")
    sonnet35 = ModelGetter("claude-3-5-sonnet-20240620")
    haiku3 = ModelGetter("claude-3-haiku-20240307")
    # /models

    def __init__(self, api_key: str) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return await self._complete(
            request=request,
            response=CompletionResponse(),
            messages=self._messages(request),
        )
    
    async def construct[T: BaseModel](
        self,
        schema: type[T],
        request: CompletionRequest,
    ) -> StructuredOutputResponse[T]:
        return await self._construct(
            schema=schema,
            request=request,
            response=StructuredOutputResponse[T](),
            messages=self._messages(request),
        )
    
    def _messages(self, request: CompletionRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if request.history:
            for interaction in request.history.interactions:
                messages.append(self._user(interaction.user, interaction.files))
                if interaction.calls:
                    messages.append(self._tool_call(interaction.calls))
                    messages.append(self._tool_results(interaction.calls))
                messages.append(self._assistant(interaction.assistant))
        messages.append(self._user(request.user_message, request.files))
        return messages
    
    def _user(self, text: str | None, files: list[File]) -> MessageParam:
        images: list[File] = []
        pdfs: list[File] = []
        for file in files:
            if file.mimetype.startswith("image/"):
                images.append(file)
            elif file.mimetype == "application/pdf":
                pdfs.append(file)
            else:
                raise ValueError(f"{self} can't handle {file} (only images and PDFs are supported)")
        content: str | list[dict[str, Any]]
        if images or pdfs:
            content = []
            if text:
                content.append({"type": "text", "text": text})
            for image in images:
                if image.data:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image.mimetype,
                            "data": image.base64,
                        },
                    })
                else:
                    content.append({
                        "type": "image",
                        "source": {"type": "url", "url": image.url},
                    })
            for pdf in pdfs:
                if pdf.data:
                    content.append({
                        "type": "pdf",
                        "source": {
                            "type": "base64",
                            "media_type": pdf.mimetype,
                            "data": pdf.base64,
                        },
                    })
                else:
                    content.append({
                        "type": "pdf",
                        "source": {"type": "url", "url": pdf.url},
                    })
        else:
            content = text or CompletionRequest.no_content
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
                    "content": call.return_value_string,
                }
                for call in calls
            ]
        )

    async def _complete(
        self,
        request: CompletionRequest,
        response: CompletionResponse,
        messages: list[MessageParam],
    ) -> CompletionResponse:
        completion = await self.client.messages.create(
            model=request.model,
            messages=messages,
            max_tokens=request.max_tokens or self.max_tokens,
            system=_optional(request.system_prompt),
            temperature=_optional(request.temperature),
            top_p=_optional(request.top_p),
            top_k=_optional(request.top_k),
            stop_sequences=_optional(request.stop),
            tools=self._tools(request),
            tool_choice=self._tool_choice(request),
        )
        text: list[str] = []
        tool_calls: list[ContentBlock] = []
        for content in completion.content:
            if content.type == "tool_use":
                tool_calls.append(content)
            if content.type == "text":
                text.append(content.text)
        if tool_calls:
            await self._run_tools(request, response, messages, tool_calls)
            return await self._complete(request, response, messages)
        if not text:
            raise ValueError(f"no text in response from {self}")
        response.contents.append("\n".join(text))
        return response
    
    async def _construct[T: BaseModel](
        self,
        schema: type[T],
        request: CompletionRequest,
        response: StructuredOutputResponse[T],
        messages: list[MessageParam],
    ) -> StructuredOutputResponse[T]:
        request.tools["structured_output"] = ToolSchema({
            "name": "structured_output",
            "description": "Use this tool to output a structured object",
            "parameters": schema.model_json_schema(),
        })
        request.append_instruction("""
            Use the structured_output tool to output a structured object.
        """)
        completion = await self.client.messages.create(
            model=request.model,
            messages=messages,
            max_tokens=request.max_tokens or self.max_tokens,
            system=_optional(request.system_prompt),
            temperature=_optional(request.temperature),
            top_p=_optional(request.top_p),
            top_k=_optional(request.top_k),
            stop_sequences=_optional(request.stop),
            tools=self._tools(request),
            tool_choice=self._tool_choice(request),
        )
        outputs: list[dict[str, Any]] = []
        tool_calls: list[ContentBlock] = []
        for content in completion.content:
            if content.type == "tool_use":
                if content.name == "structured_output":
                    outputs.append(content.input)
                else:
                    tool_calls.append(content)
        if tool_calls:
            await self._run_tools(request, response, messages, tool_calls)
            return await self._construct(schema, request, response, messages)
        if not outputs:
            raise ValueError(f"no outputs in response from {self}")
        for output in outputs:
            response.outputs.append(schema.model_validate(output))
        return response

    def _tools(self, request: CompletionRequest) -> list[ToolParam] | anthropic.NotGiven:
        if not request.tools:
            return anthropic.NOT_GIVEN
        tools: list[ToolParam] = []
        for tool in request.tools.values():
            tools.append({
                "name": tool.schema["name"],
                "description": tool.schema["description"],
                "input_schema": tool.schema["parameters"],
            })
        return tools
    
    def _tool_choice(
        self,
        request: CompletionRequest,
    ) -> anthropic.NotGiven | Literal["none"] | Literal["any"] | ToolChoiceToolParam:
        if request.use_tool is None:
            return anthropic.NOT_GIVEN
        if request.use_tool is False:
            return ToolChoiceToolParam(type="none")
        if request.use_tool is True:
            return ToolChoiceToolParam(type="any")
        return ToolChoiceToolParam(type="tool", name=request.use_tool, disable_parallel_tool_use=True)

    async def _run_tools(
        self,
        request: Request,
        response: Response,
        messages: list[dict[str, str]],
        tool_calls: list[ToolUseBlock],
    ) -> None:
        calls: list[Call] = []
        for tool_call in tool_calls:
            call = Call(tool_call.id, request.tools[tool_call.name], tool_call.input)
            calls.append(call)
        await Call.async_run_all(calls)
        messages.append(self._tool_call(calls))
        messages.append(self._tool_results(calls))
        response.calls.extend(calls)
    

def _optional[T](value: T | None) -> T | anthropic.NotGiven:
    return value if value is not None else anthropic.NOT_GIVEN