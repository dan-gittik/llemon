from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, AsyncIterator, Literal, cast

import openai
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
    ParsedChoice,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as StreamChoice
from openai.types.chat.chat_completion_content_part_image_param import ImageURL as ImageURLParam
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_param import File as ChatcompletionContentPartFileParam
from openai.types.chat.chat_completion_content_part_param import FileFile as FileParam
from openai.types.chat.chat_completion_message_tool_call_param import Function as FunctionParam
from openai.types.completion_usage import CompletionUsage
from openai.types.shared_params import FunctionDefinition, ResponseFormatJSONObject
from pydantic import BaseModel

import llemon
from llemon.types import NS, Error, ToolCalls, ToolDeltas, ToolStream
from llemon.utils import Emoji, async_parallelize, filtered_dict

if TYPE_CHECKING:
    from llemon import (
        Call,
        ClassifyRequest,
        ClassifyResponse,
        File,
        GenerateObjectRequest,
        GenerateObjectResponse,
        GenerateRequest,
        GenerateResponse,
        GenerateStreamRequest,
        GenerateStreamResponse,
    )

FILE_IDS = "openai.file_ids"
FILE_HASHES = "openai.file_hashes"

log = logging.getLogger(__name__)


class OpenAILLM(llemon.LLMProvider):

    client: openai.AsyncOpenAI

    async def prepare_generation(self, request: GenerateRequest, state: NS) -> None:
        await super().prepare_generation(request, state)
        if isinstance(request, llemon.GenerateRequest):
            if request.files:
                log.debug("uploading files")
            await async_parallelize((self._upload_file, file, state) for file in request.files)

    async def cleanup_generation(self, state: NS) -> None:
        await async_parallelize((self._delete_file, file_id) for file_id in state.pop(FILE_IDS, set()))
        state.pop(FILE_HASHES, None)
        await super().cleanup_generation(state)

    async def count_tokens(self, request: GenerateRequest) -> int:
        raise NotImplementedError()

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return await self._generate(request, llemon.GenerateResponse(request))

    async def generate_stream(self, request: GenerateStreamRequest) -> GenerateStreamResponse:
        return await self._generate_stream(request, llemon.GenerateStreamResponse(request))

    async def generate_object[T: BaseModel](self, request: GenerateObjectRequest[T]) -> GenerateObjectResponse[T]:
        if not request.llm.config.supports_structured_output:
            return await self._generate_json(request, llemon.GenerateObjectResponse(request))
        return await self._generate_object(request, llemon.GenerateObjectResponse(request))

    async def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        if not request.use_logit_biasing:
            return await super().classify(request)
        response = llemon.ClassifyResponse(request)
        log.debug("classifying with logit biasing")
        tokens = [str(i) for i in range(len(request.answers))]
        token_ids = await request.llm.tokenizer.encode(*tokens)
        if len(token_ids) != len(request.answers):
            raise request.error(f"can't do classification with {len(request.answers)} answers")
        logit_bias = {str(token_id): 100 for token_id in token_ids}
        generate_response = await self._generate(request, llemon.GenerateResponse(request), logit_bias=logit_bias)
        answer_num = int(generate_response.text)
        answer = request.answers[answer_num]
        log.debug("classification: %s", answer)
        response.complete_answer(answer)
        return response

    async def _upload_file(self, file: File, state: NS) -> None:
        if not file.data and file.is_image:
            log.debug("%s is an image URL; skipping", file)
            return
        await file.fetch()
        assert file.data is not None
        hash: File | None = state.get(FILE_HASHES, {}).get(file.md5)
        if hash:
            log.debug("%s is already uploaded as %s; reusing", file, hash.id)
            file.id = hash.id
            return
        file_object = await self.client.files.create(
            file=(file.name, file.data, file.mimetype),
            purpose="assistants",
        )
        log.debug("uploaded %s as %s", file.name, file_object.id)
        file.id = file_object.id
        state.setdefault(FILE_IDS, set()).add(file_object.id)
        state.setdefault(FILE_HASHES, {})[file.md5] = file

    async def _delete_file(self, file_id: str) -> None:
        log.debug("deleting %s", file_id)
        await self.client.files.delete(file_id)

    async def _messages(self, request: GenerateRequest) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        if request.instructions:
            instructions = await request.render_instructions()
            log.debug(Emoji.SYSTEM + "%s", instructions)
            messages.append(self._system(instructions))
        if request.history:
            self._log_history(request.history)
            for request_, response_ in request.history:
                if not isinstance(request_, llemon.GenerateRequest):
                    continue
                assert isinstance(response_, llemon.GenerateResponse)
                messages.append(await self._user(request_.user_input, request_.files))
                if response_.calls:
                    messages.append(self._tool_call(response_.calls))
                    messages.extend(self._tool_results(response_.calls))
                messages.append(self._assistant(response_.text))
        log.debug(Emoji.USER + "%s", request.user_input)
        messages.append(await self._user(request.user_input, request.files))
        return messages

    def _system(self, prompt: str) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(role="system", content=prompt)

    async def _user(self, text: str, files: list[File]) -> ChatCompletionUserMessageParam:
        content: str | list[ChatCompletionContentPartParam]
        if files:
            content = []
            if text:
                content.append(
                    ChatCompletionContentPartTextParam(
                        type="text",
                        text=text,
                    )
                )
            for file in files:
                if file.is_image:
                    content.append(
                        ChatCompletionContentPartImageParam(
                            type="image_url",
                            image_url=ImageURLParam(url=file.url),
                        )
                    )
                else:
                    if not file.id:
                        raise Error(f"{file} was not uploaded to {self}")
                    content.append(
                        ChatcompletionContentPartFileParam(
                            type="file",
                            file=FileParam(file_id=file.id),
                        )
                    )
        else:
            content = text
        return ChatCompletionUserMessageParam(role="user", content=content)

    def _assistant(self, content: str) -> ChatCompletionAssistantMessageParam:
        return ChatCompletionAssistantMessageParam(role="assistant", content=content)

    def _tool_call(self, calls: list[Call]) -> ChatCompletionAssistantMessageParam:
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCallParam(
                    type="function",
                    id=call.id,
                    function=FunctionParam(name=call.tool.compatible_name, arguments=call.arguments_json),
                )
                for call in calls
            ],
        )

    def _tool_results(self, calls: list[Call]) -> list[ChatCompletionToolMessageParam]:
        return [
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=call.id,
                content=call.result_json,
            )
            for call in calls
        ]

    async def _generate(
        self,
        request: GenerateRequest,
        response: GenerateResponse,
        messages: list[ChatCompletionMessageParam] | None = None,
        json: bool = False,
        logit_bias: dict[str, int] | openai.NotGiven = openai.NOT_GIVEN,
    ) -> GenerateResponse:
        if messages is None:
            messages = await self._messages(request)
        try:
            extra_body = filtered_dict(
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                min_p=request.min_p,
            )
            response_format = ResponseFormatJSONObject(type="json_object") if json else openai.NOT_GIVEN
            openai_response = await self.with_overrides(self.client.chat.completions.create)(
                request,
                model=request.llm.model,
                messages=messages,
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
                temperature=request.temperature,
                max_completion_tokens=request.max_tokens,
                n=request.variants,
                seed=request.seed,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                top_p=request.top_p,
                stop=request.stop,
                extra_body=extra_body,
                response_format=response_format,
                logit_bias=logit_bias,
                timeout=request.timeout,
            )
        except openai.APIError as error:
            raise request.error(str(error))
        request.id = openai_response.id
        self._set_usage(response, openai_response.usage)
        result, is_tool = self._parse_choices(request, openai_response.choices)
        if is_tool:
            await self._run_tools(request, response, messages, cast(ToolCalls, result))
            return await self._generate(request, response, messages=messages, json=json)
        result = cast(list[str], result)
        for variant in result:
            log.debug(Emoji.ASSISTANT + "%s", variant)
        response.complete_text(*result)
        return response

    async def _generate_stream(
        self,
        request: GenerateStreamRequest,
        response: GenerateStreamResponse,
        messages: list[ChatCompletionMessageParam] | None = None,
    ) -> GenerateStreamResponse:
        if messages is None:
            messages = await self._messages(request)
        try:
            extra_body = filtered_dict(
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                min_p=request.min_p,
            )
            openai_response = await self.with_overrides(self.client.chat.completions.create)(
                request,
                model=request.llm.model,
                messages=messages,
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
                temperature=request.temperature,
                max_completion_tokens=request.max_tokens,
                seed=request.seed,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                top_p=request.top_p,
                stop=request.stop,
                extra_body=extra_body,
                stream=True,
                stream_options={"include_usage": True},
                timeout=request.timeout,
            )
        except openai.APIError as error:
            raise request.error(str(error))

        async def stream() -> AsyncIterator[str]:
            tool_stream: ToolStream = {}
            async for chunk in openai_response:
                if not chunk.choices:
                    request.id = chunk.id
                    self._set_usage(response, chunk.usage)
                    break
                result, is_tool = self._parse_stream_choices(request, chunk.choices)
                if is_tool:
                    for index, id, name, arguments in cast(ToolDeltas, result):
                        if index not in tool_stream:
                            tool_stream[index] = (id, name, [])
                        else:
                            tool_stream[index][2].append(arguments)
                elif result:
                    yield cast(str, result)
            if tool_stream:
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
        messages: list[ChatCompletionMessageParam] | None = None,
    ) -> GenerateObjectResponse[T]:
        if messages is None:
            messages = await self._messages(request)
        try:
            extra_body = filtered_dict(
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                min_p=request.min_p,
            )
            openai_response = await self.with_overrides(self.client.beta.chat.completions.parse)(
                request,
                model=request.llm.model,
                messages=messages,
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
                temperature=request.temperature,
                max_completion_tokens=request.max_tokens,
                n=request.variants,
                seed=request.seed,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                top_p=request.top_p,
                stop=request.stop,
                extra_body=extra_body,
                response_format=request.schema,
                timeout=request.timeout,
            )
        except openai.APIError as error:
            raise request.error(str(error))
        request.id = openai_response.id
        self._set_usage(response, openai_response.usage)
        result, is_tool = self._parse_object_choices(request, openai_response.choices)
        if is_tool:
            await self._run_tools(request, response, messages, cast(ToolCalls, result))
            return await self._generate_object(request, response, messages=messages)
        result = cast(list[T], result)
        for variant in result:
            log.debug(Emoji.ASSISTANT + "%s", variant)
        response.complete_object(*result)
        return response

    async def _generate_json[T: BaseModel](
        self,
        request: GenerateObjectRequest[T],
        response: GenerateObjectResponse[T],
    ) -> GenerateObjectResponse[T]:
        log.debug("%s doesn't support structured output; using JSON instead", request.llm.model)
        request.append_json_instruction()
        generate_response = await self._generate(request, llemon.GenerateResponse(request), json=True)
        data = request.schema.model_validate_json(generate_response.text)
        response.complete_object(data)
        response.calls = generate_response.calls
        return response

    def _tools(self, request: GenerateRequest) -> list[ChatCompletionToolParam] | openai.NotGiven:
        if not request.tools:
            return openai.NOT_GIVEN
        tools: list[ChatCompletionToolParam] = []
        for tool in request.tools_dict.values():
            tools.append(
                ChatCompletionToolParam(
                    type="function",
                    function=FunctionDefinition(
                        name=tool.compatible_name,
                        description=tool.description,
                        parameters=tool.parameters,
                        strict=True,
                    ),
                )
            )
        return tools

    def _tool_choice(
        self,
        request: GenerateRequest,
    ) -> openai.NotGiven | Literal["none"] | Literal["required"] | ChatCompletionNamedToolChoiceParam:
        if request.use_tool is None:
            return openai.NOT_GIVEN
        if request.use_tool is False:
            return "none"
        if request.use_tool is True:
            return "required"
        return ChatCompletionNamedToolChoiceParam(
            type="function",
            function={"name": request.get_tool_name(request.use_tool)},
        )

    def _parse_choices(
        self,
        request: GenerateRequest,
        choices: list[Choice],
    ) -> tuple[list[str], Literal[False]] | tuple[ToolCalls, Literal[True]]:
        if not choices:
            raise request.error(f"{request} has no response")
        results: list[str] = []
        for choice in choices:
            self._check_finish_reason(request, choice)
            tools = self._choice_tools(choice)
            if tools:
                return tools, True
            if not choice.message.content:
                raise request.error(f"{request} response has no content")
            results.append(choice.message.content)
        return results, False

    def _parse_object_choices[T: BaseModel](
        self,
        request: GenerateObjectRequest[T],
        choices: list[ParsedChoice[T]],
    ) -> tuple[list[T], Literal[False]] | tuple[ToolCalls, Literal[True]]:
        if not choices:
            raise request.error(f"{request} has no response")
        results: list[T] = []
        for choice in choices:
            self._check_finish_reason(request, choice)
            tools = self._choice_tools(choice)
            if tools:
                return tools, True
            if not choice.message.parsed:
                raise request.error(f"{request} response has no data")
            results.append(choice.message.parsed)
        return results, False

    def _parse_stream_choices(
        self,
        request: GenerateStreamRequest,
        choices: list[StreamChoice],
    ) -> tuple[str, Literal[False]] | tuple[ToolDeltas, Literal[True]]:
        if not choices:
            raise request.error(f"{request} has no response")
        choice = choices[0]
        self._check_finish_reason(request, choice)
        if choice.delta.tool_calls:
            tool_deltas: ToolDeltas = []
            if choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    if not tool_call.function:
                        continue
                    tool_deltas.append(
                        (
                            tool_call.index,
                            tool_call.id or "",
                            tool_call.function.name or "",
                            tool_call.function.arguments or "",
                        )
                    )
            return tool_deltas, True
        text = choice.delta.content or ""
        return text, False

    def _check_finish_reason(self, request: GenerateRequest, choice: Choice | StreamChoice) -> None:
        match choice.finish_reason:
            case "stop":
                if isinstance(choice, Choice):
                    refusal = choice.message.refusal
                else:
                    refusal = choice.delta.refusal
                if refusal:
                    raise request.error(f"{request} was refused: {refusal}")
            case "length":
                if not request.return_incomplete_message:
                    raise request.error(f"{request} response exceeded the maximum length")
            case "tool_calls" | "function_call":
                pass
            case "content_filter":
                raise request.error(f"{request} was blocked")

    def _choice_tools(self, choice: Choice | ParsedChoice) -> ToolCalls | None:
        tool_calls: ToolCalls = []
        if choice.message.tool_calls:
            for tool in choice.message.tool_calls:
                if tool.type != "function":
                    continue
                tool_calls.append((tool.id, tool.function.name, json.loads(tool.function.arguments)))
        return tool_calls

    async def _run_tools(
        self,
        request: GenerateRequest,
        response: GenerateResponse,
        messages: list[ChatCompletionMessageParam],
        tools: ToolCalls,
    ) -> None:
        calls = [Call(id, request.tools_dict[name], args) for id, name, args in tools]
        await async_parallelize(call.run for call in calls)
        messages.append(self._tool_call(calls))
        messages.extend(self._tool_results(calls))
        response.calls.extend(calls)

    def _set_usage(self, response: GenerateResponse, usage: CompletionUsage | None) -> None:
        if usage is None:
            return
        response.input_tokens += usage.prompt_tokens or 0
        if usage.prompt_tokens_details:
            cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
            response.input_tokens -= cached_tokens
            response.cache_tokens += cached_tokens
        response.output_tokens += usage.completion_tokens or 0
        if usage.completion_tokens_details:
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
            response.output_tokens -= reasoning_tokens
            response.reasoning_tokens += reasoning_tokens