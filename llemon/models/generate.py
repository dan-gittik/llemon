from __future__ import annotations

import json
import logging
from functools import cached_property
from typing import ClassVar, Self

from pydantic import BaseModel

from llemon.errors import ConfigurationError
from llemon.models.file import File
from llemon.models.request import Request, Response
from llemon.models.tool import Call, Tool, load_tool, resolve_tools
from llemon.types import NS, FilesArgument, History, RenderArgument, ToolsArgument
from llemon.utils.concat import concat
from llemon.utils.logs import ASSISTANT, FILE, TOOL, USER
from llemon.utils.rendering import Rendering
from llemon.utils.trim import trim

log = logging.getLogger(__name__)


class GenerateRequest(Request):

    no_content: ClassVar[str] = "."
    return_incomplete_messages: ClassVar[bool] = False

    def __init__(
        self,
        *,
        model: LLMModel,
        history: History | None = None,
        instructions: str | None = None,
        user_input: str | None = None,
        context: NS | None = None,
        render: RenderArgument | None = None,
        files: FilesArgument = None,
        tools: ToolsArgument | None = None,
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
        prediction: str | NS | BaseModel | None = None,
        return_incomplete_message: bool | None = None,
    ) -> None:
        super().__init__(history=history)
        if instructions is not None:
            instructions = trim(instructions)
        if user_input is not None:
            user_input = trim(user_input)
        if context is None:
            context = {}
        self.model = model
        self.instructions = instructions
        self.context = context
        self.rendering = Rendering.resolve(render)
        self.files = File.resolve(files)
        self.tools = resolve_tools(tools)
        self.use_tool = use_tool
        self.variants = variants
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.stop = stop
        self.prediction = self._resolve_prediction(prediction)
        self._user_input = user_input
        self._instructions: str | None = None
        self._return_incomplete_message = return_incomplete_message
        self.context.update(request=self)

    def __str__(self) -> str:
        return f"{self.model}({self.user_input!r})"

    @cached_property
    def user_input(self) -> str:
        if not self._user_input:
            if self.files:
                return ""
            return self.no_content
        return self._user_input

    @cached_property
    def tools_dict(self) -> dict[str, Tool]:
        tools_dict: dict[str, Tool] = {}
        for tool in self.tools:
            if isinstance(tool, Tool):
                tools_dict[tool.compatible_name] = tool
            else:
                for tool in tool.tools:
                    tools_dict[tool.compatible_name] = tool
        return tools_dict

    @cached_property
    def return_incomplete_message(self) -> bool:
        if self._return_incomplete_message is None:
            return self.return_incomplete_messages
        return self._return_incomplete_message

    def check_supported(self) -> None:
        if self.variants is not None and not self.model.config.supports_variants:
            raise ConfigurationError(f"{self.model} doesn't support multiple responses")
        if self.tools and not self.model.config.supports_tools:
            raise ConfigurationError(f"{self.model} doesn't support tools")
        for file in self.files:
            if not self.model.config.accepts_files:
                raise ConfigurationError(f"{self.model} doesn't support files")
            if file.mimetype not in self.model.config.accepts_files:
                raise ConfigurationError(f"{self.model} doesn't support {file.mimetype} files ({file})")

    def append_instruction(self, instruction: str) -> None:
        instruction = trim(instruction)
        if not self.instructions:
            self.instructions = instruction
        else:
            self.instructions += "\n" + instruction
        self._instructions = None

    async def render_instructions(self) -> str:
        if self._instructions is None:
            if not self.instructions:
                self._instructions = ""
            elif self.rendering:
                self._instructions = await self.rendering.render(self.instructions, self.context)
            else:
                self._instructions = self.instructions
        return self._instructions

    def get_tool_name(self, tool: str) -> str:
        if tool not in self.tools_dict:
            raise ValueError(f"tool {tool!r} not found (available tools are {concat(self.tools_dict)})")
        return self.tools_dict[tool].compatible_name

    def format(self, emoji: bool = True) -> str:
        output: list[str] = []
        user = USER if emoji else "User: "
        output.append(f"{user}{self.user_input}")
        file = FILE if emoji else "File: "
        for file in self.files:
            output.append(f"{file}{file.name}")
        return "\n".join(output)

    def dump(self) -> NS:
        data = super().dump()
        data.update(
            model=self.model.dump(),
            instructions=self._instructions,
            user_input=self._user_input,
            context=self.context or None,
            render=self.rendering.bracket if self.rendering else None,
            files=[file.dump() for file in self.files],
            tools=[tool.dump() for tool in self.tools],
            use_tool=self.use_tool,
            variants=self.variants,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            top_p=self.top_p,
            top_k=self.top_k,
            stop=self.stop,
            prediction=self.prediction,
            return_incomplete_message=self.return_incomplete_message,
        )
        data = {key: value for key, value in data.items() if value is not None}
        return data

    @classmethod
    def _restore(self, data: NS) -> tuple[NS, NS]:
        args, attrs = super()._restore(data)
        args.update(
            model=LLMModel.load(data["model"]),
            instructions=data.get("instructions"),
            user_input=data.get("user_input"),
            context=data.get("context"),
            render=Rendering.resolve(data.get("render")),
            files=[File.load(file) for file in data.get("files", [])],
            tools=[load_tool(tool) for tool in data.get("tools", [])],
            use_tool=data.get("use_tool"),
            variants=data.get("variants"),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            seed=data.get("seed"),
            frequency_penalty=data.get("frequency_penalty"),
            presence_penalty=data.get("presence_penalty"),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            stop=data.get("stop"),
            prediction=data.get("prediction"),
            return_incomplete_message=data.get("return_incomplete_message"),
        )
        return args, attrs

    def _resolve_prediction(self, prediction: str | NS | BaseModel | None) -> str | None:
        if prediction is None:
            return None
        if isinstance(prediction, BaseModel):
            return prediction.model_dump_json()
        try:
            return json.dumps(prediction)
        except TypeError:
            return str(prediction)

    def _copy(self) -> Self:
        request = super()._copy()
        request.files = [file._copy() for file in self.files]
        return request


class GenerateResponse(Response):

    request: GenerateRequest

    def __init__(self, request: GenerateRequest) -> None:
        super().__init__(request)
        self.calls: list[Call] = []
        self._texts: list[str] = []
        self._selected = 0

    def __str__(self) -> str:
        return f"{self.request.model}: {self.text}"

    @cached_property
    def texts(self) -> list[str]:
        if not self.ended:
            raise self._incomplete_request()
        return self._texts

    @cached_property
    def text(self) -> str:
        return self._texts[self._selected]

    def dump(self) -> NS:
        data = super().dump()
        data.update(
            calls=[call.dump() for call in self.calls],
            texts=self._texts,
            selected=self._selected,
        )
        return data

    def complete_text(self, *texts: str) -> None:
        self._texts = [text.strip() for text in texts]
        super().complete()

    def select(self, index: int) -> None:
        if index < 0 or index >= len(self._texts):
            raise ValueError(f"invalid index {index} (response has {len(self._texts)} variants)")
        self._selected = index
        self.__dict__.pop("text", None)

    def format(self, emoji: bool = True) -> str:
        output: list[str] = []
        tool = TOOL if emoji else "Tool: "
        for call in self.calls:
            result = call.result["error"] if "error" in call.result else call.result["return_value"]
            output.append(f"{tool}{call.signature} -> {result}")
        assistant = ASSISTANT if emoji else "Assistant: "
        output.append(f"{assistant}{self.text}")
        return "\n".join(output)

    @classmethod
    def _restore(self, data: NS) -> tuple[NS, NS]:
        args, attrs = super()._restore(data)
        attrs.update(
            calls=[Call.load(call) for call in data["calls"]],
            _texts=data["texts"],
            _selected=data["selected"],
        )
        return args, attrs


from llemon.apis.llm.llm_model import LLMModel
