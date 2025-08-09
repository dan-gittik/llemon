from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai.types import (
    Content,
    Part,
    UserContent,
    ModelContent,
    GenerateContentConfig,
    HttpOptions,
    Tool,
    ToolListUnion,
    FunctionDeclaration,
    FunctionCall,
    AutomaticFunctionCallingConfig,
)
from pydantic import BaseModel

from ..file import File
from ..llm import LLM
from ..model import ModelGetter
from ..requests import CompletionRequest, CompletionResponse, Request, Response, StructuredOutputResponse
from ..tool import Call


class Gemini(LLM):

    # models
    pro25 = ModelGetter("gemini-2.5-pro")
    flash25 = ModelGetter("gemini-2.5-flash")
    lite25 = ModelGetter("gemini-2.5-flash-lite")
    flash2 = ModelGetter("gemini-2.0-flash")
    lite2 = ModelGetter("gemini-2.0-flash-lite")
    # /models

    def __init__(
        self,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        version: str | None = None,
    ) -> None:
        if sum([bool(api_key), bool(project) or bool(location)]) != 1:
            raise ValueError("either API key or project and location must be provided")
        options: dict[str, Any] = {}
        if version:
            options["http_options"] = HttpOptions(api_version=version)
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client(project=project, location=location, vertexai=True)
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return await self._complete(
            request=request,
            response=CompletionResponse(),
            config=self._config(request),
            contents=await self._contents(request),
        )
    
    async def construct[T: BaseModel](self, schema: type[T], request: CompletionRequest) -> StructuredOutputResponse[T]:
        config = self._config(request, schema)
        contents = await self._contents(request)
        response = StructuredOutputResponse[T]()
        await self._construct(request, response, config, contents, schema)
        return response

    def _config(
        self,
        request: CompletionRequest,
        schema: type[BaseModel] | None = None,
    ) -> GenerateContentConfig:
        config = GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            seed=request.seed,
            candidate_count=request.n,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=request.stop,
            tools=self._tools(request),
            automatic_function_calling=AutomaticFunctionCallingConfig(
                disable=True,
            ),
        )
        if request.system_prompt:
            config.system_instruction = self._system(request.system_prompt)
        if schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = schema
        return config
    
    async def _contents(self, request: CompletionRequest) -> list[Content]:
        contents: list[Content] = []
        if request.history:
            for interaction in request.history.interactions:
                contents.append(await self._user(interaction.user, interaction.files))
                if interaction.calls:
                    contents.append(self._tool_call(interaction.calls))
                    contents.append(self._tool_results(interaction.calls))
                contents.append(self._assistant(interaction.assistant))
        contents.append(await self._user(request.user_content, request.files))
        return contents
    
    def _system(self, prompt: str) -> Content:
        return Content(parts=[Part.from_text(text=prompt)])
    
    async def _user(self, text: str | None, files: list[File]) -> UserContent:
        parts: list[Part] = []
        for file in files:
            if not file.data:
                await file.fetch()
            part = Part.from_bytes(data=file.data, mime_type=file.mimetype)
            parts.append(part)
        if text:
            parts.append(Part.from_text(text=text))
        elif not parts:
            parts.append(Part.from_text(text=CompletionRequest.no_content))
        return UserContent(parts=parts)
    
    def _assistant(self, content: str) -> ModelContent:
        return ModelContent(parts=[Part.from_text(text=content)])
    
    def _tool_call(self, calls: list[Call]) -> ModelContent:
        parts: list[Part] = []
        for call in calls:
            part = Part.from_function_call(
                name=call.tool.name,
                args=call.arguments,
            )
            parts.append(part)
        return ModelContent(parts=parts)
    
    def _tool_results(self, calls: list[Call]) -> Content:
        parts: list[Part] = []
        for call in calls:
            parts.append(Part.from_function_response(
                name=call.tool.name,
                response={"result": call.return_value_string},
            ))
        return Content(role="tool", parts=parts)
    
    def _tools(self, request: CompletionRequest) -> ToolListUnion | None:
        if not request.tools or request.use_tool is False:
            return None
        tools: ToolListUnion = []
        for tool in request.tools.values():
            tools.append(Tool(
                function_declarations=[FunctionDeclaration(
                    name=tool.schema["name"],
                    description=tool.schema["description"],
                    parameters_json_schema=tool.schema["parameters"],
                )]
            ))
        return tools

    async def _complete(
        self,
        request: CompletionRequest,
        response: CompletionResponse,
        config: GenerateContentConfig,
        contents: list[Content],
    ) -> None:
        completion = await self.client.aio.models.generate_content(
            model=request.model,
            contents=contents,
            config=config,
        )
        if completion.function_calls:
            await self._run_tools(request, response, contents, completion.function_calls)
            return await self._complete(request, response, config, contents)
        for candidate in completion.candidates:
            response.contents.append(candidate.content.parts[0].text)
        return response

    async def _construct[T: BaseModel](
        self,
        request: CompletionRequest,
        response: StructuredOutputResponse[T],
        config: GenerateContentConfig,
        contents: list[Content],
        schema: type[T],
    ) -> StructuredOutputResponse[T]:
        construction = await self.client.aio.models.generate_content(
            model=request.model,
            contents=contents,
            config=config,
        )
        if construction.function_calls:
            await self._run_tools(request, response, contents, construction.function_calls)
            return await self._construct(request, response, config, contents, schema)
        for candidate in construction.candidates:
            response.outputs.append(schema.model_validate_json(candidate.content.parts[0].text))
        return response

    async def _run_tools(
        self,
        request: Request,
        response: Response,
        contents: list[Content],
        function_calls: list[FunctionCall],
    ) -> None:
        calls: list[Call] = []
        for function_call in function_calls:
            call = Call(function_call.id, request.tools[function_call.name], function_call.args or {})
            calls.append(call)
        await Call.async_run_all(calls)
        contents.append(self._tool_call(calls))
        contents.append(self._tool_results(calls))
        response.calls.extend(calls)