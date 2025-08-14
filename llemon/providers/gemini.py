from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

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
from ..llm import LLM, LLMModelGetter
from ..protocol import Completion, StructuredOutput, Stream, LLMOperation
from ..tool import Call
from ..utils import USER, ASSISTANT, Error, SetupError

log = logging.getLogger(__name__)


class Gemini(LLM):

    pro25 = LLMModelGetter("gemini-2.5-pro")
    flash25 = LLMModelGetter("gemini-2.5-flash")
    lite25 = LLMModelGetter("gemini-2.5-flash-lite")
    flash2 = LLMModelGetter("gemini-2.0-flash")
    lite2 = LLMModelGetter("gemini-2.0-flash-lite")

    def __init__(
        self,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        version: str | None = None,
    ) -> None:
        if sum([bool(api_key), bool(project) or bool(location)]) != 1:
            raise SetupError("either API key or project and location must be provided")
        options: dict[str, Any] = {}
        if version:
            options["http_options"] = HttpOptions(api_version=version)
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client(project=project, location=location, vertexai=True)
    
    async def complete(self, completion: Completion) -> None:
        contents = await self._contents(completion)
        await self._complete(completion, self._config(completion), contents)
    
    async def stream(self, stream: Stream) -> None:
        contents = await self._contents(stream)
        await self._stream(stream, self._config(stream), contents)
    
    async def construct[T: BaseModel](self, structured_output: StructuredOutput[T]) -> None:
        contents = await self._contents(structured_output)
        await self._construct(structured_output, self._config(structured_output), contents)

    def _config(self, completion: Completion) -> GenerateContentConfig:
        config = GenerateContentConfig(
            temperature=completion.temperature,
            max_output_tokens=completion.max_tokens,
            seed=completion.seed,
            candidate_count=completion.num_responses,
            frequency_penalty=completion.frequency_penalty,
            presence_penalty=completion.presence_penalty,
            top_p=completion.top_p,
            top_k=completion.top_k,
            stop_sequences=completion.stop,
            tools=self._tools(completion),
            automatic_function_calling=AutomaticFunctionCallingConfig(
                disable=True,
            ),
        )
        if completion.system_prompt:
            config.system_instruction = self._system(completion.get_system_prompt())
        if isinstance(completion, StructuredOutput):
            config.response_mime_type = "application/json"
            config.response_schema = completion.schema
        return config
    
    async def _contents(self, completion: Completion) -> list[Content]:
        contents: list[Content] = []
        if completion.history:
            completion.history.log()
            for interaction in completion.history.interactions:
                contents.append(await self._user(interaction.user, interaction.files))
                if interaction.calls:
                    contents.append(self._tool_call(interaction.calls))
                    contents.append(self._tool_results(interaction.calls))
                contents.append(self._assistant(interaction.assistant))
        user_message = completion.get_user_message()
        log.debug(USER + "%s", user_message)
        contents.append(await self._user(user_message, completion.files))
        return contents
    
    def _system(self, prompt: str) -> Content:
        return Content(parts=[Part.from_text(text=prompt)])
    
    async def _user(self, text: str, files: list[File]) -> UserContent:
        parts: list[Part] = []
        if files:
            if text:
                parts.append(Part.from_text(text=text))
            for file in files:
                await file.fetch()
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
                response={"result": call.result},
            ))
        return Content(role="tool", parts=parts)
    
    def _tools(self, operation: LLMOperation) -> ToolListUnion | None:
        if not operation.tools or operation.use_tool is False:
            return None
        tools: ToolListUnion = []
        for tool in operation.tools.values():
            tools.append(Tool(
                function_declarations=[FunctionDeclaration(
                    name=tool.schema["name"],
                    description=tool.schema["description"],
                    parameters_json_schema=tool.schema["parameters"],
                )]
            ))
        return tools

    async def _complete(self, completion: Completion, config: GenerateContentConfig, contents: list[Content]) -> None:
        try:
            response = await self.client.aio.models.generate_content(
                model=completion.model.name,
                contents=contents,
                config=config,
            )
        except Exception as error:
            raise Error(error)
        if response.function_calls:
            await self._run_tools(completion, contents, response.function_calls)
            return await self._complete(completion, config, contents)
        for candidate in response.candidates:
            text = candidate.content.parts[0].text
            completion.add_text(text)
            log.debug(ASSISTANT + "%s", text)
    
    async def _stream(self, stream: Stream, config: GenerateContentConfig, contents: list[Content]) -> None:
        try:
            response = await self.client.aio.models.generate_content_stream(
                model=stream.model.name,
                contents=contents,
                config=config,
            )
        except Exception as error:
            raise Error(error)

        async def chunks() -> AsyncIterator[str]:
            function_calls: list[FunctionCall] = []
            async for chunk in response:
                if chunk.function_calls:
                    function_calls.extend(chunk.function_calls)
                elif chunk.candidates and chunk.candidates[0].content.parts:
                    text = chunk.candidates[0].content.parts[0].text
                    yield text
            if function_calls:
                await self._run_tools(stream, contents, function_calls)
                await self._stream(stream, config, contents)
                async for chunk in stream.stream:
                    yield chunk
            else:
                log.debug(ASSISTANT + "%s", stream.text)

        stream.stream = chunks()

    async def _construct[T: BaseModel](
        self,
        structured_output: StructuredOutput[T],
        config: GenerateContentConfig,
        contents: list[Content],
    ) -> None:
        try:
            response = await self.client.aio.models.generate_content(
                model=structured_output.model.name,
                contents=contents,
                config=config,
            )
        except Exception as error:
            raise Error(error)
        if response.function_calls:
            await self._run_tools(structured_output, contents, response.function_calls)
            return await self._construct(structured_output, config, contents)
        for candidate in response.candidates:
            object_ = json.loads(candidate.content.parts[0].text)
            structured_output.add_object(object_)
            log.debug(ASSISTANT + "%s", object_)

    async def _run_tools(
        self,
        operation: LLMOperation,
        contents: list[Content],
        function_calls: list[FunctionCall],
    ) -> None:
        calls: list[Call] = []
        for function_call in function_calls:
            call = Call(function_call.id, operation.tools[function_call.name], function_call.args or {})
            calls.append(call)
        await Call.async_run_all(calls)
        contents.append(self._tool_call(calls))
        contents.append(self._tool_results(calls))
        operation.calls.extend(calls)