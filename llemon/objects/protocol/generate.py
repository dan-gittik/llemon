from __future__ import annotations

import logging
import warnings
from decimal import Decimal
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

import llemon
from llemon.types import NS, FilesArgument, HistoryArgument, RenderArgument, ToolsArgument, Warning
from llemon.utils import Emoji, concat, filtered_dict, trim

if TYPE_CHECKING:
    from llemon import LLM, Call, Tool
    from llemon.objects.serializeable import DumpRefs, LoadRefs, Unpacker


log = logging.getLogger(__name__)


class GenerateRequest(llemon.Request):

    no_content: ClassVar[str] = "."
    return_incomplete_messages: ClassVar[bool] = False

    def __init__(
        self,
        *,
        llm: LLM,
        instructions: str | None = None,
        user_input: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        history: HistoryArgument = None,
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
        min_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
        return_incomplete_message: bool | None = None,
        cache: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> None:
        super().__init__(provider_options)
        if instructions is not None:
            instructions = trim(instructions)
        if user_input is not None:
            user_input = trim(user_input)
        if context is None:
            context = {}
        if history is None:
            history = []
        super().__init__(self._overrides(llm.provider, provider_options))
        self.llm = llm
        self.instructions = instructions
        self.context = context
        self.rendering = llemon.Rendering.resolve(render)
        self.history = history
        self.files = [llemon.File.resolve(file) for file in files] if files else []
        self.tools = llemon.Tool.resolve(tools)
        self.use_tool = use_tool
        self.variants = variants
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        self.stop = stop
        self.prediction = prediction
        self.cache = cache
        self.timeout = timeout
        self._user_input = user_input
        self._instructions: str | None = None
        self._return_incomplete_message = return_incomplete_message

    def __str__(self) -> str:
        return f"{self.llm}.generate({self.user_input!r})"

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
            if isinstance(tool, llemon.Tool):
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
        for parameter in self.llm.config.unsupported_parameters or []:
            if getattr(self, parameter, None) is not None:
                warnings.warn(f"{self.llm} doesn't support {parameter}", Warning)
                setattr(self, parameter, None)
        if self.tools and not self.llm.config.supports_tools:
            warnings.warn(f"{self.llm} doesn't support tools", Warning)
            self.tools = []
        accepted_files = []
        for file in self.files:
            if not self.llm.config.accepts_files:
                warnings.warn(f"{self.llm} doesn't support files", Warning)
                break
            if file.mimetype not in self.llm.config.accepts_files:
                warnings.warn(f"{self.llm} doesn't support {file.mimetype} files ({file})", Warning)
            else:
                accepted_files.append(file)
        self.files = accepted_files

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
                context = self.context | {"request": self}
                self._instructions = await self.rendering.render(self.instructions, context)
            else:
                self._instructions = self.instructions
        return self._instructions

    def get_tool_name(self, tool: str) -> str:
        if tool not in self.tools_dict:
            raise ValueError(f"tool {tool!r} not found (available tools are {concat(self.tools_dict)})")
        return self.tools_dict[tool].compatible_name

    def format(self, emoji: bool = True) -> str:
        output: list[str] = []
        user = Emoji.USER if emoji else "User: "
        output.append(f"{user}{self.user_input}")
        file_ = Emoji.FILE if emoji else "File: "
        for file in self.files:
            output.append(f"{file_}{file.display_name}")
        return "\n".join(output)

    @classmethod
    def _restore(cls, unpacker: Unpacker, refs: LoadRefs) -> tuple[NS, NS]:
        args, attrs = super()._restore(unpacker, refs)
        args.update(
            llm=refs.get_llm(unpacker.get("llm", str)),
            instructions=unpacker.get("instructions", str, None),
            user_input=unpacker.get("user_input", str, None),
            context=unpacker.get("context", dict, None),
            render=llemon.Rendering.resolve(unpacker.get("render", (bool, str), None)),
            history=refs.get_history(unpacker.get("history", list, [])),
            files=[refs.get_file(name) for name in unpacker.get("files", list, [])],
            tools=[refs.get_tool(name) for name in unpacker.get("tools", list, [])],
            use_tool=unpacker.get("use_tools", (bool, str), None),
            variants=unpacker.get("variants", int, None),
            temperature=unpacker.get("temperature", float, None),
            max_tokens=unpacker.get("max_tokens", int, None),
            seed=unpacker.get("seed", int, None),
            frequency_penalty=unpacker.get("frequency_penalty", float, None),
            presence_penalty=unpacker.get("presence_penalty", float, None),
            top_p=unpacker.get("top_p", float, None),
            min_p=unpacker.get("min_p", float, None),
            top_k=unpacker.get("top_k", int, None),
            stop=unpacker.get("stop", list, None),
            prediction=unpacker.get("prediction", str, None),
            return_incomplete_message=unpacker.get("return_incomplete_message", bool, None),
            cache=unpacker.get("cache", bool, None),
            timeout=unpacker.get("timeout", float, None),
        )
        return args, attrs

    def _dump(self, refs: DumpRefs) -> NS:
        data = super()._dump(refs)
        refs.add_llm(self.llm)
        for request, response in self.history:
            refs.add_request(request)
            refs.add_response(response)
        for file in self.files:
            refs.add_file(file)
        for tool in self.tools:
            refs.add_tool(tool)
        data.update(
            filtered_dict(
                llm=self.llm.model,
                instructions=self.instructions,
                user_input=self.user_input,
                context=self.context or None,
                render=self.rendering.bracket if self.rendering else None,
                history=[request.id for request, _ in self.history],
                files=[file.name for file in self.files],
                tools=[tool.name for tool in self.tools],
                use_tool=self.use_tool,
                variants=self.variants,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                top_p=self.top_p,
                min_p=self.min_p,
                top_k=self.top_k,
                stop=self.stop,
                prediction=self.prediction,
                return_incomplete_message=self.return_incomplete_message,
                cache=self.cache,
                timeout=self.timeout,
            )
        )
        return data


class GenerateResponse(llemon.Response):

    request: GenerateRequest

    def __init__(self, request: GenerateRequest) -> None:
        super().__init__(request)
        self.calls: list[Call] = []
        self.input_tokens = 0
        self.cache_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self._texts: list[str] = []
        self._selected = 0

    def __str__(self) -> str:
        return f"{self.request.llm}: {self.text}"

    @cached_property
    def texts(self) -> list[str]:
        if not self.ended:
            raise self._incomplete_request()
        return self._texts

    @cached_property
    def text(self) -> str:
        return self._texts[self._selected]

    @cached_property
    def cost(self) -> Decimal:
        return (
            Decimal(self.input_tokens) * Decimal(self.request.llm.config.cost_per_1m_input_tokens or 0)
            + Decimal(self.cache_tokens) * Decimal(self.request.llm.config.cost_per_1m_cache_tokens or 0)
            + Decimal(self.output_tokens) * Decimal(self.request.llm.config.cost_per_1m_output_tokens or 0)
            + Decimal(self.reasoning_tokens) * Decimal(self.request.llm.config.cost_per_1m_output_tokens or 0)
        ) / 1_000_000

    @cached_property
    def tps(self) -> float:
        return (self.reasoning_tokens + self.output_tokens) / self.duration

    def complete_text(self, *texts: str) -> None:
        self._texts = [text.strip() for text in texts]
        self.complete()

    def select(self, index: int) -> None:
        if index < 0 or index >= len(self._texts):
            raise ValueError(f"invalid index {index} (response has {len(self._texts)} variants)")
        self._selected = index
        self.__dict__.pop("text", None)

    def format(self, emoji: bool = True) -> str:
        output: list[str] = []
        tool = Emoji.TOOL if emoji else "Tool: "
        for call in self.calls:
            result = call.result["error"] if "error" in call.result else call.result["return_value"]
            output.append(f"{tool}{call.signature} -> {result}")
        assistant = Emoji.ASSISTANT if emoji else "Assistant: "
        output.append(f"{assistant}{self.text}")
        return "\n".join(output)

    def _restore(self, unpacker: Unpacker, refs: LoadRefs) -> None:
        self.calls = [llemon.Call._load(call, refs) for call in unpacker.load_list("calls", required=False)]
        self.input_tokens = unpacker.get("input_tokens", int)
        self.input_tokens = unpacker.get("input_tokens", int)
        self.cache_tokens = unpacker.get("cache_tokens", int)
        self.output_tokens = unpacker.get("output_tokens", int)
        self._texts = unpacker.get("texts", list)
        self._selected = unpacker.get("selected", int)

    def _dump(self, refs: DumpRefs) -> NS:
        data = super()._dump(refs)
        data.update(
            filtered_dict(
                calls=[call._dump(refs) for call in self.calls],
                input_tokens=self.input_tokens,
                cache_tokens=self.cache_tokens,
                output_tokens=self.output_tokens,
                reasoning_tokens=self.reasoning_tokens,
                texts=self._texts,
                selected=self._selected,
            )
        )
        return data
