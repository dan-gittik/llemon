from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Literal, cast, overload

import openai
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletion,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolParam,
    ParsedChatCompletion,
    ParsedChoice,
)
from openai.types.shared_params import FunctionDefinition
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
    File as ChatcompletionContentPartFileParam,
)
from pydantic import BaseModel

from ..file import File
from ..llm import LLM, LLMModelGetter
from ..protocol import Call, Completion, LLMOperation, StructuredOutput
from ..utils import Error, async_parallelize

FILE_IDS = "openai.file_ids"
FILE_HASHES = "openai.file_hashes"

log = logging.getLogger(__name__)


class OpenAI(LLM):

    gpt5 = LLMModelGetter("gpt-5")
    gpt5_mini = LLMModelGetter("gpt-5-mini")
    gpt5_nano = LLMModelGetter("gpt-5-nano")
    gpt41 = LLMModelGetter("gpt-4.1")
    gpt41_mini = LLMModelGetter("gpt-4.1-mini")
    gpt41_nano = LLMModelGetter("gpt-4.1-nano")
    gpt4o = LLMModelGetter("gpt-4o")
    gpt4o_mini = LLMModelGetter("gpt-4o-mini")
    gpt4 = LLMModelGetter("gpt-4")
    gpt4_turbo = LLMModelGetter("gpt-4-turbo")
    gpt35_turbo = LLMModelGetter("gpt-3.5-turbo")

    def __init__(self, api_key: str) -> None:
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def setup(self, operation: LLMOperation, state: dict[str, Any]) -> None:
        await super().setup(operation, state)
        log.debug("uploading files")
        for file in operation.files:
            if not file.data and file.is_image:
                log.debug("%s is an image URL; skipping", file)
                continue
            await file.fetch()
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
    
    async def teardown(self, state: dict[str, Any]) -> None:
        await async_parallelize([(self._delete_file, (file_id,), {}) for file_id in state.pop(FILE_IDS, set())])
        state.pop(FILE_HASHES, None)
        await super().teardown(state)
    
    async def complete(self, completion: Completion) -> None:
        await self._complete(completion, await self._messages(completion))
    
    async def construct[T: BaseModel](self, structured_output: StructuredOutput[T]) -> None:
        if not structured_output.model.config.supports_structured_output:
            log.debug("%s doesn't support structured output; using JSON instead", structured_output.model)
            structured_output.append_json_instruction()
            await self.complete(structured_output)
            for text in structured_output.texts:
                log.debug("decoding %s as JSON", text)
                structured_output.add_object(json.loads(text))
            return
        await self._construct(structured_output, await self._messages(structured_output))
    
    async def _messages(self, completion: Completion) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if completion.system_prompt:
            messages.append(self._system(completion.get_system_prompt()))
        if completion.history:
            for interaction in completion.history.interactions:
                messages.append(await self._user(interaction.user, interaction.files))
                if interaction.calls:
                    messages.append(self._tool_call(interaction.calls))
                    messages.extend(self._tool_results(interaction.calls))
                messages.append(self._assistant(interaction.assistant))
        messages.append(await self._user(completion.get_user_message(), completion.files))
        return messages
    
    def _system(self, prompt: str) -> ChatCompletionSystemMessageParam:
        log.debug("SYSTEM: %s", prompt)
        return ChatCompletionSystemMessageParam(role="system", content=prompt)
    
    async def _user(self, text: str, files: list[File]) -> ChatCompletionUserMessageParam:
        log.debug("USER: %s", text)
        for file in files:
            log.debug("FILE: %s", file)
        content: str | list[dict[str, Any]]
        if files:
            content = []
            if text:
                content.append(ChatCompletionContentPartTextParam(
                    type="text",
                    text=text,
                ))
            for file in files:
                if file.is_image:
                    content.append(ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url={"url": file.url},
                    ))
                else:
                    content.append(ChatcompletionContentPartFileParam(
                        type="file",
                        file={"file_id": file.id},
                    ))
        else:
            content = text
        return ChatCompletionUserMessageParam(role="user", content=content)
    
    def _assistant(self, content: str) -> ChatCompletionAssistantMessageParam:
        log.debug("ASSISTANT: %s", content)
        return ChatCompletionAssistantMessageParam(role="assistant", content=content)
    
    def _tool_call(self, calls: list[Call]) -> ChatCompletionAssistantMessageParam:
        for call in calls:
            log.debug("TOOL CALL: %s", call)
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            tool_calls=[
                {
                    "type": "function",
                    "id": call.id,
                    "function": {"name": call.tool.name, "arguments": call.arguments_json},
                }
                for call in calls
            ]
        )

    def _tool_results(self, calls: list[Call]) -> list[ChatCompletionToolMessageParam]:
        for call in calls:
            log.debug("TOOL RESULT: %s", call.result_json)
        return [
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=call.id,
                content=call.result_json,
            )
            for call in calls
        ]
    
    async def _complete(self, completion: Completion, messages: list[ChatCompletionMessageParam]) -> None:
        try:
            response_format = {"type": "json_object"} if isinstance(completion, StructuredOutput) else openai.NOT_GIVEN
            response = await self.client.chat.completions.create(
                model=completion.model.name,
                messages=messages,
                tools=self._tools(completion),
                tool_choice=self._tool_choice(completion),
                temperature=_optional(completion.temperature),
                max_tokens=_optional(completion.max_tokens),
                seed=_optional(completion.seed),
                frequency_penalty=_optional(completion.frequency_penalty),
                presence_penalty=_optional(completion.presence_penalty),
                n=_optional(completion.num_responses),
                top_p=_optional(completion.top_p),
                stop=_optional(completion.stop),
                response_format=response_format,
            )
        except openai.APIError as error:
            raise Error(error)
        for choice in self._choices(response):
            if choice.finish_reason == "tool_calls":
                await self._run_tools(completion, messages, choice.message.tool_calls)
                return await self._complete(completion, messages)
            if not choice.message.content:
                raise Error(f"no content in response from {self}")
            completion.add_text(choice.message.content)

    async def _construct[T: BaseModel](
        self,
        structured_output: StructuredOutput[T],
        messages: list[ChatCompletionMessageParam],
    ) -> None:
        try:
            response = await self.client.beta.chat.completions.parse(
                model=structured_output.model.name,
                messages=messages,
                tools=self._tools(structured_output),
                tool_choice=self._tool_choice(structured_output),
                temperature=_optional(structured_output.temperature),
                max_tokens=_optional(structured_output.max_tokens),
                seed=_optional(structured_output.seed),
                frequency_penalty=_optional(structured_output.frequency_penalty),
                presence_penalty=_optional(structured_output.presence_penalty),
                n=_optional(structured_output.num_responses),
                top_p=_optional(structured_output.top_p),
                stop=_optional(structured_output.stop),
                response_format=structured_output.schema,
            )
        except openai.APIError as error:
            raise Error(error)
        for choice in self._choices(response):
            if choice.finish_reason == "tool_calls":
                await self._run_tools(structured_output, messages, choice.message.tool_calls)
                return await self._construct(structured_output, messages)
            structured_output.add_object(cast(T, choice.message.parsed))
    
    def _tools(self, completion: Completion) -> list[ChatCompletionToolParam] | openai.NotGiven:
        if not completion.tools:
            return openai.NOT_GIVEN
        tools: list[ChatCompletionToolParam] = []
        for tool in completion.tools.values():
            tools.append(ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=tool.schema["name"],
                    description=tool.schema["description"],
                    parameters=tool.schema["parameters"],
                    strict=True,
                ),
            ))
        return tools
    
    def _tool_choice(
        self,
        completion: Completion,
    ) -> openai.NotGiven | Literal["none"] | Literal["required"] | ChatCompletionNamedToolChoiceParam:
        if completion.use_tool is None:
            return openai.NOT_GIVEN
        if completion.use_tool is False:
            return "none"
        if completion.use_tool is True:
            return "required"
        return ChatCompletionNamedToolChoiceParam(
            type="function",
            function={"name": completion.use_tool},
        )

    @overload
    def _choices(self, response: ParsedChatCompletion) -> Iterator[ParsedChoice]: ...

    @overload
    def _choices(self, response: ChatCompletion) -> Iterator[Choice]: ...

    def _choices(self, response: ChatCompletion | ParsedChatCompletion) -> Iterator[Choice | ParsedChoice]:
        if not response.choices:
            raise Error(f"no response from {self}")
        for choice in response.choices:
            if choice.finish_reason == "content_filter":
                raise Error(f"response from {self} was filtered")
            if choice.finish_reason == "length":
                raise Error(f"response from {self} exceeded the maximum length")
            if choice.message.refusal:
                raise Error(f"response from {self} was refused: {choice.message.refusal}")
            yield choice

    async def _run_tools(
        self,
        completion: Completion,
        messages: list[dict[str, str]],
        tool_calls: list[ChatCompletionMessageToolCall],
    ) -> None:
        calls: list[Call] = []
        for tool_call in tool_calls:
            args = json.loads(tool_call.function.arguments)
            call = Call(tool_call.id, completion.tools[tool_call.function.name], args)
            calls.append(call)
        await Call.async_run_all(calls)
        messages.append(self._tool_call(calls))
        messages.extend(self._tool_results(calls))
        completion.calls.extend(calls)
    
    async def _delete_file(self, file_id: str) -> None:
        log.debug("deleting %s", file_id)
        await self.client.files.delete(file_id)


def _optional[T](value: T | None) -> T | openai.NotGiven:
    return value if value is not None else openai.NOT_GIVEN