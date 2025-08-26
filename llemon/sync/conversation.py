from __future__ import annotations

import copy
import re
import warnings
from types import TracebackType
from typing import Iterator, cast

from pydantic import BaseModel

from llemon.objects.file import File
from llemon.objects.tool import resolve_tools
from llemon.sync.rendering import Rendering
from llemon.sync.types import NS, Error, FilesArgument, History, RenderArgument, ToolsArgument
from llemon.utils import parallelize, schema_to_model


SPACES = re.compile(r"\s+")


class Conversation:

    def __init__(
        self,
        model: LLMModel,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        tools: ToolsArgument = None,
        history: History | None = None,
    ) -> None:
        if history is None:
            history = []
        if context is None:
            context = {}
        if tools is None:
            tools = []
        self.finished = False
        self.model = model
        self.instructions = instructions
        self.context = context
        self.rendering = Rendering.resolve(render)
        self.tools = resolve_tools(tools)
        self.history = history
        self._state: NS = {}

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return f"<conversation: {self.format(one_line=True)}>"

    def __bool__(self) -> bool:
        return bool(self.history)

    def __len__(self) -> int:
        return len(self.history)

    def __iter__(self) -> Iterator[tuple[Request, Response]]:
        yield from self.history

    def __getitem__(self, index: int | slice) -> Conversation:
        if isinstance(index, int):
            history = [self.history[index]]
        else:
            history = self.history[index]
        return self.replace(history=history)

    def __aenter__(self) -> Conversation:
        return self

    def __aexit__(
        self,
        exception: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.finish()

    def __del__(self) -> None:
        if not self.finished:
            warnings.warn(f"{self!r} was never finished", Warning)

    @classmethod
    def load(cls, data: NS) -> Conversation:
        return load(Conversation, data)

    @property
    def llm(self) -> LLM:
        return self.model.llm

    def dump(self) -> NS:
        return dump(self)

    def replace(
        self,
        model: LLMModel | None = None,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        tools: ToolsArgument = None,
        history: History | None = None,
    ) -> Conversation:
        return self.__class__(
            model=model or self.model,
            instructions=instructions or self.instructions,
            context=context or self.context.copy(),
            render=render or self.rendering,
            tools=tools or self.tools,
            history=history or self.history,
        )

    def prepare(self) -> Conversation:
        self._assert_not_finished()
        self.history = self._copy_history()
        parallelize([(self.llm.prepare, (request, self._state), {}) for request, _ in self])
        return self

    def finish(self, cleanup: bool = True) -> None:
        self._assert_not_finished()
        if cleanup:
            self.llm.cleanup(self._state)
        self.finished = True

    def render_instructions(self) -> str:
        if self.instructions is None:
            return ""
        if self.rendering:
            return self.rendering.render(self.instructions, self.context)
        return self.instructions

    def format(self, one_line: bool = False, emoji: bool = True) -> str:
        interactions: list[str] = []
        for request, response in self.history:
            interactions.append(request.format(emoji=emoji))
            interactions.append(response.format(emoji=emoji))
        if one_line:
            interactions = [SPACES.sub(" ", interaction) for interaction in interactions]
            separator = " | "
        else:
            separator = "\n"
        return separator.join(interactions)

    def generate(
        self,
        message: str | None = None,
        *,
        save: bool = True,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        variants: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
        return_incomplete_message: bool | None = None,
    ) -> GenerateResponse:
        self._assert_not_finished()
        request = GenerateRequest(
            model=self.model,
            instructions=instructions or self.instructions,
            user_input=message,
            context=self.context | (context or {}),
            render=render or self.rendering,
            history=self.history,
            files=files,
            tools=[*self.tools, *(tools or [])],
            use_tool=use_tool,
            variants=variants,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            prediction=prediction,
            return_incomplete_message=return_incomplete_message,
        )
        self.llm.prepare(request, self._state)
        response = self.llm.generate(request)
        if save:
            self.history.append((request, response))
        return response

    def generate_stream(
        self,
        message: str | None = None,
        *,
        save: bool = True,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
        return_incomplete_message: bool | None = None,
    ) -> GenerateStreamResponse:
        self._assert_not_finished()
        request = GenerateStreamRequest(
            model=self.model,
            instructions=instructions or self.instructions,
            user_input=message,
            context=self.context | (context or {}),
            render=render or self.rendering,
            history=self.history,
            files=files,
            tools=[*self.tools, *(tools or [])],
            use_tool=use_tool,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            prediction=prediction,
            return_incomplete_message=return_incomplete_message,
        )
        self.llm.prepare(request, self._state)
        response = self.llm.generate_stream(request)
        if save:
            self.history.append((request, response))
        return response

    def generate_object[T: BaseModel](
        self,
        schema: type[T] | NS,
        message: str | None = None,
        *,
        save: bool = True,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        prediction: T | NS | None = None,
    ) -> GenerateObjectResponse[T]:
        self._assert_not_finished()
        if isinstance(schema, dict):
            model_class = cast(type[T], schema_to_model(schema))
        else:
            model_class = schema
        request = GenerateObjectRequest(
            schema=model_class,
            model=self.model,
            instructions=instructions or self.instructions,
            user_input=message,
            context=self.context | (context or {}),
            render=render or self.rendering,
            history=self.history,
            files=files,
            tools=[*self.tools, *(tools or [])],
            use_tool=use_tool,
            temperature=temperature,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            prediction=prediction,
        )
        self.llm.prepare(request, self._state)
        response = self.llm.generate_object(request)
        if save:
            self.history.append((request, response))
        return response

    def classify(
        self,
        question: str,
        answers: list[str] | type[bool],
        user_input: str,
        *,
        save: bool = True,
        reasoning: bool = False,
        null_answer: bool = True,
        context: NS | None = None,
        render: RenderArgument = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
    ) -> ClassifyResponse:
        self._assert_not_finished()
        request = ClassifyRequest(
            model=self.model,
            question=question,
            answers=answers,
            user_input=user_input,
            reasoning=reasoning,
            null_answer=null_answer,
            context=self.context | (context or {}),
            render=render or self.rendering,
            files=files,
            tools=[*self.tools, *(tools or [])],
            use_tool=use_tool,
        )
        self.llm.prepare(request, self._state)
        response = self.llm.classify(request)
        if save:
            self.history.append((request, response))
        return response

    def _assert_not_finished(self) -> None:
        if self.finished:
            raise Error(f"{self!r} has already finished")

    def _copy_history(self) -> list[tuple[Request, Response]]:
        history = []
        for request, response in self.history:
            if isinstance(request, GenerateRequest):
                request = copy.copy(request)
                request.id = None
                files: list[File] = []
                for file in request.files:
                    file = copy.copy(file)
                    file.id = None
                    files.append(file)
                request.files = files
            history.append((request, response))
        return history


from llemon.sync.llm import LLM
from llemon.sync.llm_model import LLMModel
from llemon.sync.classify import ClassifyRequest, ClassifyResponse
from llemon.sync.generate import GenerateRequest, GenerateResponse
from llemon.sync.generate_object import GenerateObjectRequest, GenerateObjectResponse
from llemon.sync.generate_stream import GenerateStreamRequest, GenerateStreamResponse
from llemon.objects.request import Request, Response
from llemon.sync.serialization import dump, load