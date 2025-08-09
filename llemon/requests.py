from __future__ import annotations

import json
from typing import Any, AsyncIterator, ClassVar

from pydantic import BaseModel

from .file import File
from .interaction import History
from .tool import Call, Tool
from .types import FilesArgument, HistoryArgument, ToolsArgument
from .utils import trim


class Request:

    no_content: ClassVar[str] = "."

    def __init__(
        self,
        model: str,
        user_message: str | None = None,
        history: HistoryArgument | None = None,
        files: FilesArgument = None,
        tools: ToolsArgument | None = None,
        use_tool: bool | str | None = None,
    ) -> None:
        self.model = model
        self.user_message = trim(user_message) if user_message else None
        self.history = History.resolve(history)
        self.files = File.resolve(files)
        self.tools = Tool.resolve(tools)
        self.use_tool = use_tool
    
    @property
    def user_content(self) -> str:
        if self.user_message is None:
            return self.no_content
        return self.user_message
    
    def append_instruction(self, instruction: str) -> None:
        instruction = trim(instruction)
        if not self.user_message:
            self.user_message = instruction
        else:
            self.user_message += "\n" + instruction
    

class CompletionRequest(Request):

    def __init__(
        self,
        model: str,
        user_message: str | None = None,
        system_prompt: str  | None = None,
        history: HistoryArgument | None = None,
        files: FilesArgument = None,
        tools: ToolsArgument | None = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        n: int | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        stop: list[str] | None = None,
        prediction: str | dict[str, Any] | BaseModel |None = None,
    ) -> None:
        super().__init__(model, user_message, history, files, tools, use_tool)
        self.system_prompt = trim(system_prompt) if system_prompt else None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.n = n
        self.top_p = top_p
        self.top_k = top_k
        self.stop = stop
        self.prediction = self._resolve_prediction(prediction)

    def _resolve_prediction(self, prediction: str | dict[str, Any] | BaseModel | None) -> str | None:
        if prediction is None:
            return None
        if isinstance(prediction, BaseModel):
            return prediction.model_dump_json()
        try:
            return json.dumps(prediction)
        except TypeError:
            return str(prediction)


class Response:

    def __init__(self) -> None:
        self.calls: list[Call] = []


class CompletionResponse(Response):

    def __init__(self) -> None:
        super().__init__()
        self.contents: list[str] = []


class StreamResponse(Response):

    def __init__(self, stream: AsyncIterator[str]) -> None:
        super().__init__()
        self.stream = stream
        self.chunks: list[str] = []
        self.ttft: float | None = None
        
    @property
    def output(self) -> str:
        return "".join(self.chunks)
    
    async def __aiter__(self) -> AsyncIterator[str]:
        async for chunk in self.stream:
            self.chunks.append(chunk)
            yield chunk


class StructuredOutputResponse[T: BaseModel](Response):

    def __init__(self) -> None:
        super().__init__()
        self.outputs: list[T] = []


class ClassificationRequest(Request):

    def __init__(
        self,
        model: str,
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


class ClassificationResponse(Response):

    def __init__(self) -> None:
        super().__init__()
        self.answer: str = ""