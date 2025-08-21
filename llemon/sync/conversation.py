from __future__ import annotations

import re
import warnings
from types import TracebackType
from typing import TYPE_CHECKING, Iterator, cast

from pydantic import BaseModel

from llemon.errors import FinishedError
from llemon.sync.generate import GenerateRequest, GenerateResponse
from llemon.sync.generate_object import GenerateObjectRequest, GenerateObjectResponse
from llemon.sync.generate_stream import GenerateStreamRequest, GenerateStreamResponse
from llemon.models.request import Request, Response
from llemon.models.tool import Tool, Toolbox, load_tool, resolve_tools
from llemon.sync.types import NS, FilesArgument, History, RenderArgument, ToolsArgument
from llemon.utils.parallelize import parallelize
from llemon.sync.rendering import Rendering
from llemon.utils.schema import schema_to_model

if TYPE_CHECKING:
    from llemon.sync.llm import LLM

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
        history: History = []
        models: dict[str, NS] = data.get("models", {})
        files: dict[str, NS] = data.get("files", {})
        tools: dict[str, Tool | Toolbox] = {}
        for tool in data.get("tools", []):
            loaded = load_tool(tool)
            if isinstance(loaded, Toolbox):
                for tool in loaded.tools:
                    tools[tool.name] = tool
            else:
                tools[loaded.name] = loaded
        conversation = data["conversation"]
        for interaction in conversation["history"]:
            request_data = interaction["request"]
            if "model" in request_data:
                request_data["model"] = models[request_data["model"]]
            if "files" in request_data:
                request_data["files"] = [files[name] for name in request_data["files"]]
            if "tools" in request_data:
                request_data["tools"] = [tools[name] for name in request_data["tools"]]
            request = Request.load(request_data)
            response_data = interaction["response"]
            for call in response_data.get("calls", []):
                call["tool"] = tools[call["tool"]]
            response_data["request"] = request
            response = Response.load(response_data)
            response.request.history = history
            history.append((response.request, response))
        conversation_model = data["models"][conversation["model"]]
        conversation_tools = [tools[name] for name in conversation.get("tools", [])]
        return cls(
            model=LLMModel.load(conversation_model),
            instructions=conversation.get("instructions"),
            context=conversation.get("context"),
            render=Rendering.resolve(conversation.get("render")),
            tools=conversation_tools,
            history=history,
        )

    @property
    def llm(self) -> LLM:
        return self.model.llm
    
    def dump(self) -> NS:
        models = {self.model.name: self.model.dump()}
        tools = {tool.name: tool.dump() for tool in self.tools}
        files: dict[str, NS] = {}
        history: list[NS] = []
        for _, response in self.history:
            response_data = response.dump()
            for call in response_data.get("calls", []):
                call["tool"] = call["tool"]["name"]
            request_data = response_data.pop("request")
            if "model" in request_data:
                model_name = request_data["model"]["name"]
                models[model_name] = request_data["model"]
                request_data["model"] = model_name
            if "files" in request_data:
                files.update({file["name"]: file for file in request_data["files"]})
                request_data["files"] = [file["name"] for file in request_data["files"]]
            if "tools" in request_data:
                tools.update({tool["name"]: tool for tool in request_data["tools"]})
                request_data["tools"] = [tool["name"] for tool in request_data["tools"]]
            history.append({"request": request_data, "response": response_data})
        conversation = dict(
            model=self.model.name,
            instructions=self.instructions,
            context=self.context or None,
            render=self.rendering.bracket if self.rendering else False,
            tools=[tool.name for tool in self.tools],
            history=history,
        )
        return dict(
            conversation=conversation,
            models=models,
            tools=list(tools.values()),
            files=files,
        )

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
        self.history = [(request._copy(), response._copy()) for request, response in self.history]
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

    def format(self, one_line: bool = False) -> str:
        interactions: list[str] = []
        for request, response in self.history:
            interactions.append(request.format())
            interactions.append(response.format())
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
            top_p=top_p,
            top_k=top_k,
            prediction=prediction,
        )
        self.llm.prepare(request, self._state)
        response = self.llm.generate_object(request)
        if save:
            self.history.append((request, response))
        return response

    def _assert_not_finished(self) -> None:
        if self.finished:
            raise FinishedError(f"{self!r} has already finished")


from llemon.sync.llm_model import LLMModel