from __future__ import annotations

import json
import logging
from functools import cached_property
from typing import AsyncIterator, Literal, Sequence, cast

import openai
import tiktoken
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
from openai.types.chat.chat_completion_content_part_image_param import (
    ImageURL as ImageURLParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    File as ChatcompletionContentPartFileParam,
)
from openai.types.chat.chat_completion_content_part_param import FileFile as FileParam
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as FunctionParam,
)
from openai.types.shared_params import FunctionDefinition, ResponseFormatJSONObject
from pydantic import BaseModel

from llemon.core.llm.llm import LLM
from llemon.core.llm.llm_model import LLMModel
from llemon.core.llm.llm_model_property import LLMModelProperty
from llemon.core.llm.llm_tokenizer import LLMToken, LLMTokenizer
from llemon.errors import Error
from llemon.models.classify import ClassifyRequest, ClassifyResponse
from llemon.models.file import File
from llemon.models.generate import GenerateRequest, GenerateResponse
from llemon.models.generate_object import GenerateObjectRequest, GenerateObjectResponse
from llemon.models.generate_stream import GenerateStreamRequest, GenerateStreamResponse
from llemon.models.tool import Call
from llemon.types import NS, ToolCalls, ToolDeltas, ToolStream
from llemon.utils.logs import ASSISTANT, SYSTEM, USER
from llemon.utils.parallelize import async_parallelize

FILE_IDS = "openai.file_ids"
FILE_HASHES = "openai.file_hashes"
ENCODINGS: dict[str, tiktoken.Encoding] = {}

log = logging.getLogger(__name__)


class OpenAI(LLM):

    gpt5 = LLMModelProperty("gpt-5")
    gpt5_mini = LLMModelProperty("gpt-5-mini")
    gpt5_nano = LLMModelProperty("gpt-5-nano")
    gpt41 = LLMModelProperty("gpt-4.1")
    gpt41_mini = LLMModelProperty("gpt-4.1-mini")
    gpt41_nano = LLMModelProperty("gpt-4.1-nano")
    gpt4o = LLMModelProperty("gpt-4o")
    gpt4o_mini = LLMModelProperty("gpt-4o-mini")
    gpt4 = LLMModelProperty("gpt-4")
    gpt4_turbo = LLMModelProperty("gpt-4-turbo")
    gpt35_turbo = LLMModelProperty("gpt-3.5-turbo")

    def __init__(self, api_key: str) -> None:
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def prepare(self, request: GenerateRequest, state: NS) -> None:
        await super().prepare(request, state)
        if request.files:
            log.debug("uploading files")
        await async_parallelize([(self._upload_file, (file, state), {}) for file in request.files])

    async def cleanup(self, state: NS) -> None:
        await async_parallelize([(self._delete_file, (file_id,), {}) for file_id in state.pop(FILE_IDS, set())])
        state.pop(FILE_HASHES, None)
        await super().cleanup(state)

    def get_tokenizer(self, model: LLMModel) -> LLMTokenizer:
        if model.name not in ENCODINGS:
            ENCODINGS[model.name] = tiktoken.encoding_for_model(model.name)
        return OpenAITokenizer(ENCODINGS[model.name])

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return await self._generate(request, GenerateResponse(request))

    async def generate_stream(self, request: GenerateStreamRequest) -> GenerateStreamResponse:
        return await self._generate_stream(request, GenerateStreamResponse(request))

    async def generate_object[T: BaseModel](self, request: GenerateObjectRequest[T]) -> GenerateObjectResponse[T]:
        if not request.model.config.supports_structured_output:
            return await self._generate_json(request, GenerateObjectResponse(request))
        return await self._generate_object(request, GenerateObjectResponse(request))

    async def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        response = ClassifyResponse(request)
        reasoning: str | None = None
        if request.use_logit_biasing:
            log.debug("classifying with logit biasing")
            tokens = [str(i) for i in range(len(request.answers))]
            token_ids = await request.model.tokenizer.encode(*tokens)
            if len(token_ids) != len(request.answers):
                raise Error(f"can't do classification with {len(request.answers)} answers")
            logit_bias = {str(token_id): 100 for token_id in token_ids}
            generate_response = await self._generate(request, GenerateResponse(request), logit_bias=logit_bias)
            answer_num = int(generate_response.text)
        elif request.model.config.supports_json:
            log.debug("classifying with structured output")
            generate_object_request = request.to_object_request()
            generate_object_response = await self._generate_object(
                generate_object_request, GenerateObjectResponse(generate_object_request)
            )
            data = generate_object_response.object.model_dump()
            answer_num = cast(int, data["answer"])
            reasoning = data.get("reasoning")
        else:
            log.debug("classifying with generated text")
            generate_response = await self._generate(request, GenerateResponse(request))
            if not generate_response.text.isdigit():
                raise Error(f"{self} failed to classify {request} (answer was not a number: {generate_response.text})")
            answer_num = int(generate_response.text)
        if not 0 <= answer_num < len(request.answers):
            raise Error(f"{self} failed to classify {request} (answer number was out of range: {answer_num})")
        answer = request.answers[answer_num]
        log.debug("classification: %s (%s)", answer, reasoning or "no reasoning")
        response.complete_answer(answer, reasoning)
        return response

    async def _upload_file(self, file: File, state: NS) -> None:
        if not file.data and file.is_image:
            log.debug("%s is an image URL; skipping", file)
            return
        await file.async_fetch()
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
            log.debug(SYSTEM + "%s", instructions)
            messages.append(self._system(instructions))
        if request.history:
            self._log_history(request.history)
            for request_, response_ in request.history:
                if not isinstance(request_, GenerateRequest) or not isinstance(response_, GenerateResponse):
                    continue
                messages.append(await self._user(request_.user_input, request_.files))
                if response_.calls:
                    messages.append(self._tool_call(response_.calls))
                    messages.extend(self._tool_results(response_.calls))
                messages.append(self._assistant(response_.text))
        log.debug(USER + "%s", request.user_input)
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
            response_format = ResponseFormatJSONObject(type="json_object") if json else openai.NOT_GIVEN
            openai_response = await self.client.chat.completions.create(
                model=request.model.name,
                messages=messages,
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
                temperature=_optional(request.temperature),
                max_completion_tokens=_optional(request.max_tokens),
                n=_optional(request.variants),
                seed=_optional(request.seed),
                frequency_penalty=_optional(request.frequency_penalty),
                presence_penalty=_optional(request.presence_penalty),
                top_p=_optional(request.top_p),
                stop=_optional(request.stop),
                response_format=response_format,
                logit_bias=logit_bias,
            )
        except openai.APIError as error:
            raise Error(error)
        result, is_tool = self._parse_choices(openai_response.choices, request.return_incomplete_message)
        if is_tool:
            await self._run_tools(request, response, messages, cast(ToolCalls, result))
            return await self._generate(request, response, messages=messages, json=json)
        result = cast(list[str], result)
        for variant in result:
            log.debug(ASSISTANT + "%s", variant)
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
            openai_response = await self.client.chat.completions.create(
                model=request.model.name,
                messages=messages,
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
                temperature=_optional(request.temperature),
                max_completion_tokens=_optional(request.max_tokens),
                seed=_optional(request.seed),
                frequency_penalty=_optional(request.frequency_penalty),
                presence_penalty=_optional(request.presence_penalty),
                top_p=_optional(request.top_p),
                stop=_optional(request.stop),
                stream=True,
            )
        except openai.APIError as error:
            raise Error(error)

        async def stream() -> AsyncIterator[str]:
            tool_stream: ToolStream = {}
            async for chunk in openai_response:
                result, is_tool = self._parse_stream_choices(chunk.choices, request.return_incomplete_message)
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
                log.debug(ASSISTANT + "%s", response.text)

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
            openai_response = await self.client.beta.chat.completions.parse(
                model=request.model.name,
                messages=messages,
                tools=self._tools(request),
                tool_choice=self._tool_choice(request),
                temperature=_optional(request.temperature),
                max_completion_tokens=_optional(request.max_tokens),
                n=_optional(request.variants),
                seed=_optional(request.seed),
                frequency_penalty=_optional(request.frequency_penalty),
                presence_penalty=_optional(request.presence_penalty),
                top_p=_optional(request.top_p),
                stop=_optional(request.stop),
                response_format=request.schema,
            )
        except openai.APIError as error:
            raise Error(error)
        result, is_tool = self._parse_data_choices(openai_response.choices, return_incomplete_message=False)
        if is_tool:
            await self._run_tools(request, response, messages, cast(ToolCalls, result))
            return await self._generate_object(request, response, messages=messages)
        result = cast(list[T], result)
        for variant in result:
            log.debug(ASSISTANT + "%s", variant)
        response.complete_object(*result)
        return response

    async def _generate_json[T: BaseModel](
        self,
        request: GenerateObjectRequest[T],
        response: GenerateObjectResponse[T],
    ) -> GenerateObjectResponse[T]:
        log.debug("%s doesn't support structured output; using JSON instead", request.model)
        request.append_json_instruction()
        generate_response = await self._generate(request, GenerateResponse(request), json=True)
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
        choices: list[Choice],
        return_incomplete_message: bool,
    ) -> tuple[list[str], Literal[False]] | tuple[ToolCalls, Literal[True]]:
        if not choices:
            raise Error(f"no response from {self}")
        results: list[str] = []
        for choice in choices:
            print(choice)
            self._check_finish_reason(choice, return_incomplete_message)
            tools = self._choice_tools(choice)
            if tools:
                return tools, True
            if not choice.message.content:
                raise Error(f"no content in response from {self}")
            results.append(choice.message.content)
        return results, False

    def _parse_data_choices[T: BaseModel](
        self,
        choices: list[ParsedChoice[T]],
        return_incomplete_message: bool,
    ) -> tuple[list[T], Literal[False]] | tuple[ToolCalls, Literal[True]]:
        if not choices:
            raise Error(f"no response from {self}")
        results: list[T] = []
        for choice in choices:
            self._check_finish_reason(choice, return_incomplete_message)
            tools = self._choice_tools(choice)
            if tools:
                return tools, True
            if not choice.message.parsed:
                raise Error(f"no data in response from {self}")
            results.append(choice.message.parsed)
        return results, False

    def _parse_stream_choices(
        self,
        choices: list[StreamChoice],
        return_incomplete_message: bool,
    ) -> tuple[str, Literal[False]] | tuple[ToolDeltas, Literal[True]]:
        if not choices:
            raise Error(f"no response from {self}")
        choice = choices[0]
        self._check_finish_reason(choice, return_incomplete_message)
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

    def _check_finish_reason(self, choice: Choice | StreamChoice, return_incomplete_message: bool) -> None:
        match choice.finish_reason:
            case "stop":
                if isinstance(choice, Choice):
                    refusal = choice.message.refusal
                else:
                    refusal = choice.delta.refusal
                if refusal:
                    raise Error(f"response from {self} was blocked: {refusal}")
            case "length":
                if not return_incomplete_message:
                    raise Error(f"response from {self} exceeded the maximum length")
            case "tool_calls" | "function_call":
                pass
            case "content_filter":
                raise Error(f"response from {self} was blocked")

    def _choice_tools(self, choice: Choice | ParsedChoice) -> ToolCalls | None:
        tool_calls: ToolCalls = []
        if choice.message.tool_calls:
            for tool in choice.message.tool_calls:
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
        await Call.async_run_all(calls)
        messages.append(self._tool_call(calls))
        messages.extend(self._tool_results(calls))
        response.calls.extend(calls)


class OpenAITokenizer(LLMTokenizer):

    def __init__(self, encoding: tiktoken.Encoding) -> None:
        self.encoding = encoding

    async def count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    async def parse(self, text: str) -> Sequence[OpenAIToken]:
        tokens: list[OpenAIToken] = []
        token: OpenAIToken | None = None
        for token_id in self.encoding.encode(text):
            token = OpenAIToken(token_id, self.encoding, token)
            tokens.append(token)
        return tokens

    async def encode(self, *texts: str) -> list[int]:
        ids: list[int] = []
        for text in texts:
            ids.extend(self.encoding.encode(text))
        return ids

    async def decode(self, ids: list[int]) -> str:
        return self.encoding.decode(ids)


class OpenAIToken(LLMToken):

    def __init__(self, id: int, encoding: tiktoken.Encoding, prev: OpenAIToken | None) -> None:
        super().__init__(id)
        self._encoding = encoding
        self._prev = prev

    @cached_property
    def text(self) -> str:
        return self._encoding.decode([self.id])

    @cached_property
    def offset(self) -> int:
        if self._prev is None:
            return 0
        return self._prev.offset + len(self._prev.text)


def _optional[T](value: T | None) -> T | openai.NotGiven:
    return value if value is not None else openai.NOT_GIVEN
