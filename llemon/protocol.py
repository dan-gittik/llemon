from __future__ import annotations

import json
from functools import cached_property
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, ClassVar, override

from pydantic import BaseModel

from .file import File
from .formatting import Formatting
from .interaction import History
from .tool import Call, Tool
from .types import FilesArgument, HistoryArgument, ToolsArgument, FormattingArgument
from .utils import trim, UnsupportedError

if TYPE_CHECKING:
    from .llm import LLMModel


class LLMOperation:

    no_content: ClassVar[str] = "."

    def __init__(
        self,
        model: LLMModel,
        user_message: str | None = None,
        context: dict[str, Any] | None = None,
        formatting: FormattingArgument | None = None,
        history: HistoryArgument | None = None,
        files: FilesArgument = None,
        tools: ToolsArgument | None = None,
        use_tool: bool | str | None = None,
    ) -> None:
        self.model = model
        self.user_message = trim(user_message) if user_message else None
        self.context = context if context is not None else {}
        self.formatting = Formatting.resolve(formatting)
        self.history = History.resolve(history)
        self.files = File.resolve(files)
        self.tools = Tool.resolve(tools)
        self.use_tool = use_tool
        self.calls: list[Call] = []
        self.check_supported()
    
    @property
    def user_content(self) -> str:
        if self.user_message is None:
            return self.no_content
        return self.user_message
    
    def get_user_message(self, format: bool = True) -> str:
        if not self.user_message:
            if self.files:
                return ""
            return self.no_content
        if format and self.formatting:
            return self.formatting.format(self.user_message, self.context)
        return self.user_message

    def check_supported(
        self,
        num_responses: int | None = None,
        streaming: bool = False,
        json: bool = False,
    ) -> None:
        if self.tools and not self.model.config.supports_tools:
            raise UnsupportedError(f"{self.model} doesn't support tools")
        for file in self.files:
            if not self.model.config.accepts_files:
                raise UnsupportedError(f"{self.model} doesn't support files")
            if file.mimetype not in self.model.config.accepts_files:
                raise UnsupportedError(f"{self.model} doesn't support {file.mimetype} files ({file})")
        if num_responses and not self.model.config.supports_multiple_responses:
            raise UnsupportedError(f"{self.model} doesn't support multiple responses")
        if streaming and not self.model.config.supports_streaming:
            raise UnsupportedError(f"{self.model} doesn't support streaming")
        if json and not self.model.config.supports_json:
            raise UnsupportedError(f"{self.model} doesn't support structured output")


class Completion(LLMOperation):

    def __init__(
        self,
        model: LLMModel,
        system_prompt: str  | None = None,
        user_message: str | None = None,
        context: dict[str, Any] | None = None,
        formatting: FormattingArgument | None = None,
        history: HistoryArgument | None = None,
        files: FilesArgument = None,
        tools: ToolsArgument | None = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        num_responses: int | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        stop: list[str] | None = None,
        prediction: str | dict[str, Any] | BaseModel | None = None,
    ) -> None:
        super().__init__(model, user_message, context, formatting, history, files, tools, use_tool)
        self.check_supported(num_responses=num_responses)
        self.system_prompt = trim(system_prompt) if system_prompt else None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.num_responses = num_responses
        self.top_p = top_p
        self.top_k = top_k
        self.stop = stop
        self.prediction = self._resolve_prediction(prediction)
        self.texts: list[str] = []
    
    @property
    def text(self) -> str:
        return self.texts[0]
    
    def get_system_prompt(self, format: bool = True) -> str:
        if not self.system_prompt:
            return self.no_content
        if format and self.formatting:
            return self.formatting.format(self.system_prompt, self.context)
        return self.system_prompt
    
    def append_instruction(self, instruction: str) -> None:
        instruction = trim(instruction)
        if not self.system_prompt:
            self.system_prompt = instruction
        else:
            self.system_prompt += "\n" + instruction
        
    def add_text(self, *texts: str) -> None:
        self.texts.extend(texts)

    def _resolve_prediction(self, prediction: str | dict[str, Any] | BaseModel | None) -> str | None:
        if prediction is None:
            return None
        if isinstance(prediction, BaseModel):
            return prediction.model_dump_json()
        try:
            return json.dumps(prediction)
        except TypeError:
            return str(prediction)


class Stream(Completion):

    @override
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.check_supported(streaming=True)
        self.stream: AsyncIterator[str] | None = None
        self.chunks: list[str] = []
        self.ttft: float | None = None
        
    @property
    def text(self) -> str:
        return "".join(self.chunks)
    
    async def __aiter__(self) -> AsyncIterator[str]:
        started = time.monotonic()
        async for chunk in self.stream:
            if self.ttft is None:
                self.ttft = time.monotonic() - started
            self.chunks.append(chunk)
            yield chunk


class StructuredOutput[T: BaseModel](Completion):

    @override
    def __init__(self, *args, schema: type[T], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.check_supported(json=True)
        self.schema = schema
        self._dicts: list[dict[str, Any]] = []
        self._objects: list[T] = []
    
    @cached_property
    def dicts(self) -> list[dict[str, Any]]:
        if self._dicts:
            return self._dicts
        return [model.model_dump() for model in self._objects]
    
    @property
    def dict(self) -> dict[str, Any]:
        return self.dicts[0]
    
    @cached_property
    def objects(self) -> list[T]:
        if self._objects:
            return self._objects
        print(self._dicts)
        return [self.schema.model_validate(dict_) for dict_ in self._dicts]
    
    @cached_property
    def object(self) -> T:
        return self.objects[0]
    
    def add_object(self, *objects: T | dict[str, Any]) -> None:
        for object_ in objects:
            if isinstance(object_, dict):
                self._dicts.append(object_)
            else:
                self._objects.append(object_)
    
    def append_json_instruction(self) -> None:
        self.append_instruction(
            f"Answer ONLY in JSON that adheres EXACTLY to the following JSON schema: {self.schema.model_json_schema()}"
        )


class Classification(LLMOperation):

    def __init__(
        self,
        model: LLMModel,
        question: str,
        answers: list[str],
        user_message: str | None = None,
        history: HistoryArgument | None = None,
        files: FilesArgument = None,
        tools: ToolsArgument | None = None,
        use_tool: bool | str | None = None,
    ) -> None:
        super().__init__(model, user_message, history, files, tools, use_tool)
        self.question = question
        self.answers = answers
        self.answer: str | None = None
    
    def set_answer(self, answer: str) -> None:
        self.answer = answer