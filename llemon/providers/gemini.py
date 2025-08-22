from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Literal, cast

from google import genai
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    Content,
    FinishReason,
    FunctionDeclaration,
    FunctionCallingConfig,
    FunctionCallingConfigMode,
    GenerateContentConfig,
    GenerateContentResponse,
    HttpOptions,
    ModelContent,
    Part,
    Tool,
    ToolConfig,
    ToolListUnion,
    UserContent,
)
from pydantic import BaseModel

from llemon.apis.llm.llm import LLM
from llemon.apis.llm.llm_model_property import LLMModelProperty
from llemon.errors import ConfigurationError, Error
from llemon.models.file import File
from llemon.models.generate import GenerateRequest, GenerateResponse
from llemon.models.generate_object import GenerateObjectRequest, GenerateObjectResponse
from llemon.models.generate_stream import GenerateStreamRequest, GenerateStreamResponse
from llemon.models.tool import Call
from llemon.types import NS, ToolCalls
from llemon.utils.logs import ASSISTANT, SYSTEM, USER

log = logging.getLogger(__name__)


class Gemini(LLM):

    pro25 = LLMModelProperty("gemini-2.5-pro")
    flash25 = LLMModelProperty("gemini-2.5-flash")
    lite25 = LLMModelProperty("gemini-2.5-flash-lite")
    flash2 = LLMModelProperty("gemini-2.0-flash")
    lite2 = LLMModelProperty("gemini-2.0-flash-lite")

    def __init__(
        self,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        version: str | None = None,
    ) -> None:
        if sum([bool(api_key), bool(project) or bool(location)]) != 1:
            raise ConfigurationError("either API key or project and location must be provided")
        options: NS = {}
        if version:
            options["http_options"] = HttpOptions(api_version=version)
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client(project=project, location=location, vertexai=True)

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return await self._generate(request, GenerateResponse(request))

    async def generate_stream(self, request: GenerateStreamRequest) -> GenerateStreamResponse:
        return await self._generate_stream(request, GenerateStreamResponse(request))

    async def generate_object[T: BaseModel](self, request: GenerateObjectRequest[T]) -> GenerateObjectResponse[T]:
        return await self._generate_object(request, GenerateObjectResponse(request))

    async def _config(self, request: GenerateRequest) -> GenerateContentConfig:
        config = GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            candidate_count=request.variants,
            seed=request.seed,
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
        if request.instructions:
            config.system_instruction = self._system(await request.render_instructions())
        if request.use_tool is not None:
            if request.use_tool is False:
                function_config = FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.NONE,
                )
            elif request.use_tool is True:
                function_config = FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.ANY,
                )
            else:
                function_config = FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.ANY,
                    allowed_function_names=[request.get_tool_name(request.use_tool)],
                )
            config.tool_config = ToolConfig(function_calling_config=function_config)
        if isinstance(request, GenerateObjectRequest):
            config.response_mime_type = "application/json"
            config.response_schema = request.schema
        return config

    async def _contents(self, request: GenerateRequest) -> list[Content]:
        contents: list[Content] = []
        if request.instructions:
            log.debug(SYSTEM + "%s", await request.render_instructions())
        if request.history:
            self._log_history(request.history)
            for request_, response_ in request.history:
                if not isinstance(request_, GenerateRequest) or not isinstance(response_, GenerateResponse):
                    continue
                contents.append(await self._user(request_.user_input, request_.files))
                if response_.calls:
                    contents.append(self._tool_call(response_.calls))
                    contents.append(self._tool_results(response_.calls))
                contents.append(self._assistant(response_.text))
        log.debug(USER + "%s", request.user_input)
        contents.append(await self._user(request.user_input, request.files))
        return contents

    def _system(self, instructions: str) -> Content:
        return Content(parts=[Part.from_text(text=instructions)])

    async def _user(self, text: str, files: list[File]) -> UserContent:
        parts: list[Part] = []
        if files:
            if text:
                parts.append(Part.from_text(text=text))
            for file in files:
                await file.async_fetch()
                assert file.data is not None
                part = Part.from_bytes(data=file.data, mime_type=file.mimetype)
                parts.append(part)
        else:
            parts.append(Part.from_text(text=text))
        return UserContent(parts=parts)

    def _assistant(self, content: str) -> ModelContent:
        return ModelContent(parts=[Part.from_text(text=content)])

    def _tool_call(self, calls: list[Call]) -> ModelContent:
        parts: list[Part] = []
        for call in calls:
            part = Part.from_function_call(
                name=call.tool.compatible_name,
                args=call.arguments,
            )
            parts.append(part)
        return ModelContent(parts=parts)

    def _tool_results(self, calls: list[Call]) -> Content:
        parts: list[Part] = []
        for call in calls:
            parts.append(
                Part.from_function_response(
                    name=call.tool.compatible_name,
                    response={"result": call.result},
                )
            )
        return Content(role="tool", parts=parts)

    def _tools(self, request: GenerateRequest) -> ToolListUnion | None:
        if not request.tools or request.use_tool is False:
            return None
        tools: ToolListUnion = []
        for tool in request.tools_dict.values():
            tools.append(
                Tool(
                    function_declarations=[
                        FunctionDeclaration(
                            name=tool.compatible_name,
                            description=tool.description,
                            parameters_json_schema=tool.parameters,
                        )
                    ]
                )
            )
        return tools

    async def _generate(
        self,
        request: GenerateRequest,
        response: GenerateResponse,
        config: GenerateContentConfig | None = None,
        contents: list[Content] | None = None,
    ) -> GenerateResponse:
        if config is None:
            config = await self._config(request)
        if contents is None:
            contents = await self._contents(request)
        try:
            gemini_response = await self.client.aio.models.generate_content(
                model=request.model.name,
                contents=contents,
                config=config,
            )
        except Exception as error:
            raise Error(error)
        result, is_tool = self._parse_response(gemini_response, request.return_incomplete_message)
        if is_tool:
            await self._run_tools(request, response, contents, cast(ToolCalls, result))
            return await self._generate(request, response, config=config, contents=contents)
        result = cast(list[str], result)
        for variant in result:
            log.debug(ASSISTANT + "%s", variant)
        response.complete_text(*result)
        return response

    async def _generate_stream(
        self,
        request: GenerateStreamRequest,
        response: GenerateStreamResponse,
        config: GenerateContentConfig | None = None,
        contents: list[Content] | None = None,
    ) -> GenerateStreamResponse:
        if config is None:
            config = await self._config(request)
        if contents is None:
            contents = await self._contents(request)
        try:
            anthropic_response = await self.client.aio.models.generate_content_stream(
                model=request.model.name,
                contents=contents,
                config=config,
            )
        except Exception as error:
            raise Error(error)

        async def stream() -> AsyncIterator[str]:
            tool_calls: ToolCalls = []
            async for chunk in anthropic_response:
                result, is_tool = self._parse_response(chunk, request.return_incomplete_message)
                if is_tool:
                    tool_calls.extend(cast(ToolCalls, result))
                elif result:
                    yield cast(str, result[0])
            if tool_calls:
                await self._run_tools(request, response, contents, tool_calls)
                await self._generate_stream(request, response, config=config, contents=contents)
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
        config: GenerateContentConfig | None = None,
        contents: list[Content] | None = None,
    ) -> GenerateObjectResponse[T]:
        if config is None:
            config = await self._config(request)
        if contents is None:
            contents = await self._contents(request)
        try:
            gemini_response = await self.client.aio.models.generate_content(
                model=request.model.name,
                contents=contents,
                config=config,
            )
        except Exception as error:
            raise Error(error)
        result, is_tool = self._parse_response(gemini_response, return_incomplete_message=False)
        if is_tool:
            await self._run_tools(request, response, contents, cast(ToolCalls, result))
            return await self._generate_object(request, response, config=config, contents=contents)
        result = cast(list[str], result)
        objects = [request.schema.model_validate(json.loads(variant)) for variant in result]
        for variant in objects:
            log.debug(ASSISTANT + "%s", variant)
        response.complete_object(*objects)
        return response

    def _parse_response(
        self,
        response: GenerateContentResponse,
        return_incomplete_message: bool,
    ) -> tuple[list[str], Literal[False]] | tuple[ToolCalls, Literal[True]]:
        if not response.candidates:
            raise Error(f"{self} returned no candidates")
        for candidate in response.candidates:
            match candidate.finish_reason:
                case None | FinishReason.STOP:
                    pass
                case FinishReason.MAX_TOKENS:
                    if not return_incomplete_message:
                        raise Error(f"{self} reached the maximum number of tokens")
                case _:
                    if response.prompt_feedback and response.prompt_feedback.block_reason_message:
                        raise Error(f"{self} was blocked: {response.prompt_feedback.block_reason_message}")
                    raise Error(f"{self} was aborted: {candidate.finish_message}")
        if response.function_calls:
            tool_calls: ToolCalls = []
            for function_call in response.function_calls:
                tool_calls.append((function_call.id or "", function_call.name or "", function_call.args or {}))
            return tool_calls, True
        result: list[str] = []
        for candidate in response.candidates:
            if not candidate.content:
                raise Error(f"{self} returned no content")
            if not candidate.content.parts:
                raise Error(f"{self} returned no parts")
            if not candidate.content.parts[0].text:
                raise Error(f"{self} returned no text")
            result.append(candidate.content.parts[0].text)
        return result, False

    async def _run_tools(
        self,
        request: GenerateRequest,
        response: GenerateResponse,
        contents: list[Content],
        tool_calls: ToolCalls,
    ) -> None:
        calls = [Call(id, request.tools_dict[name], args) for id, name, args in tool_calls]
        await Call.async_run_all(calls)
        contents.append(self._tool_call(calls))
        contents.append(self._tool_results(calls))
        response.calls.extend(calls)
