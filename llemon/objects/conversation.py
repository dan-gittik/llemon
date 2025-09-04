from __future__ import annotations

import copy
import re
import warnings
from types import TracebackType
from typing import TYPE_CHECKING, Any, Iterator, Self

from pydantic import BaseModel

import llemon
from llemon.types import NS, Error, FileArgument, FilesArgument, HistoryArgument, RenderArgument, ToolsArgument
from llemon.utils import async_parallelize, filtered_dict

if TYPE_CHECKING:
    from llemon import (
        LLM,
        STT,
        TTS,
        ClassifyResponse,
        Embedder,
        EmbedResponse,
        File,
        GenerateObjectResponse,
        GenerateRequest,
        GenerateResponse,
        GenerateStreamResponse,
        Request,
        Response,
        SynthesizeResponse,
        TranscribeResponse,
    )
    from llemon.objects.serializeable import DumpRefs, LoadRefs, Unpacker

SPACES = re.compile(r"\s+")


class Conversation(llemon.Serializeable):

    def __init__(
        self,
        *,
        llm: LLM,
        stt: STT | None = None,
        tts: TTS | None = None,
        embedder: Embedder | None = None,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        history: HistoryArgument = None,
        include_messages: int | None = None,
        tools: ToolsArgument = None,
        cache: bool | None = None,
    ) -> None:
        if context is None:
            context = {}
        if history is None:
            history = []
        if tools is None:
            tools = []
        if stt is None and isinstance(llm.provider, llemon.STTProvider):
            stt = llm.provider.default_stt
        if tts is None and isinstance(llm.provider, llemon.TTSProvider):
            tts = llm.provider.default_tts
        if embedder is None and isinstance(llm.provider, llemon.EmbedderProvider):
            embedder = llm.provider.default_embedder
        self.finished = False
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.embedder = embedder
        self.instructions = instructions
        self.context = context
        self.rendering = llemon.Rendering.resolve(render)
        self.history = history
        self.include_messages = include_messages
        self.tools = llemon.Tool.resolve(tools)
        self.cache = cache
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

    async def __aenter__(self) -> Conversation:
        return self

    async def __aexit__(
        self,
        exception: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.finish()

    def __del__(self) -> None:
        if not self.finished:
            warnings.warn(f"{self!r} was never finished", Warning)

    def replace(
        self,
        llm: LLM | None = None,
        stt: STT | None = None,
        tts: TTS | None = None,
        embedder: Embedder | None = None,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        history: HistoryArgument = None,
        include_messages: int | None = None,
        tools: ToolsArgument = None,
        cache: bool | None = None,
    ) -> Conversation:
        return self.__class__(
            llm=llm or self.llm,
            stt=stt or self.stt,
            tts=tts or self.tts,
            embedder=embedder or self.embedder,
            instructions=instructions or self.instructions,
            context=context or self.context.copy(),
            render=render or self.rendering,
            history=history or self.history,
            include_messages=include_messages or self.include_messages,
            tools=tools or self.tools,
            cache=cache if cache is not None else self.cache,
        )

    async def prepare(self) -> Conversation:
        self._assert_not_finished()
        self.history = self._copy_history()
        await async_parallelize(
            (self.llm.provider.prepare_generation, request, self._state) for request, _ in self.history
        )
        return self

    async def finish(self, cleanup: bool = True) -> None:
        self._assert_not_finished()
        if cleanup:
            await self.llm.provider.cleanup_generation(self._state)
        self.finished = True

    async def render_instructions(self) -> str:
        if self.instructions is None:
            return ""
        if self.rendering:
            return await self.rendering.render(self.instructions, self.context)
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

    async def generate(
        self,
        message: str | None = None,
        *,
        save: bool = True,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        include_messages: int | None = None,
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
    ) -> GenerateResponse:
        self._assert_not_finished()
        request = llemon.GenerateRequest(
            llm=self.llm,
            instructions=instructions or self.instructions,
            user_input=message,
            context=self._resolve_context(context),
            render=render or self.rendering,
            history=self._resolve_history(include_messages),
            files=files,
            tools=self._resolve_tools(tools),
            use_tool=use_tool,
            variants=variants,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            stop=stop,
            prediction=prediction,
            return_incomplete_message=return_incomplete_message,
            cache=cache if cache is not None else self.cache,
            timeout=timeout,
            **provider_options,
        )
        await self.llm.provider.prepare_generation(request, self._state)
        response = await self.llm.provider.generate(request)
        if save:
            self.history.append((request, response))
        return response

    async def generate_stream(
        self,
        message: str | None = None,
        *,
        save: bool = True,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        include_messages: int | None = None,
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
        min_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        prediction: str | None = None,
        return_incomplete_message: bool | None = None,
        cache: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> GenerateStreamResponse:
        self._assert_not_finished()
        request = llemon.GenerateStreamRequest(
            llm=self.llm,
            instructions=instructions or self.instructions,
            user_input=message,
            context=self._resolve_context(context),
            render=render or self.rendering,
            history=self._resolve_history(include_messages),
            files=files,
            tools=self._resolve_tools(tools),
            use_tool=use_tool,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            stop=stop,
            prediction=prediction,
            return_incomplete_message=return_incomplete_message,
            cache=cache if cache is not None else self.cache,
            timeout=timeout,
            **provider_options,
        )
        await self.llm.provider.prepare_generation(request, self._state)
        response = await self.llm.provider.generate_stream(request)
        if save:
            self.history.append((request, response))
        return response

    async def generate_object[T: BaseModel](
        self,
        schema: NS | type[T],
        message: str | None = None,
        *,
        save: bool = True,
        instructions: str | None = None,
        context: NS | None = None,
        render: RenderArgument = None,
        include_messages: int | None = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        variants: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        prediction: str | NS | T | None = None,
        cache: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> GenerateObjectResponse[T]:
        self._assert_not_finished()
        request = llemon.GenerateObjectRequest(
            llm=self.llm,
            schema=schema,
            instructions=instructions or self.instructions,
            user_input=message,
            context=self._resolve_context(context),
            render=render or self.rendering,
            history=self._resolve_history(include_messages),
            files=files,
            tools=self._resolve_tools(tools),
            use_tool=use_tool,
            variants=variants,
            temperature=temperature,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            prediction=prediction,
            cache=cache if cache is not None else self.cache,
            timeout=timeout,
            **provider_options,
        )
        await self.llm.provider.prepare_generation(request, self._state)
        response = await self.llm.provider.generate_object(request)
        if save:
            self.history.append((request, response))
        return response

    async def classify(
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
        include_messages: int | None = None,
        files: FilesArgument = None,
        tools: ToolsArgument = None,
        use_tool: bool | str | None = None,
        cache: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> ClassifyResponse:
        self._assert_not_finished()
        request = llemon.ClassifyRequest(
            llm=self.llm,
            question=question,
            answers=answers,
            user_input=user_input,
            reasoning=reasoning,
            null_answer=null_answer,
            context=self._resolve_context(context),
            render=render or self.rendering,
            history=self._resolve_history(include_messages),
            files=files,
            tools=self._resolve_tools(tools),
            use_tool=use_tool,
            cache=cache if cache is not None else self.cache,
            timeout=timeout,
            **provider_options,
        )
        await self.llm.provider.prepare_generation(request, self._state)
        response = await self.llm.provider.classify(request)
        if save:
            self.history.append((request, response))
        return response

    async def embed(
        self,
        text: str,
        save: bool = True,
        **provider_options: Any,
    ) -> EmbedResponse:
        self._assert_not_finished()
        if not self.embedder:
            raise Error(f"{self!r} has no embedder associated with it")
        request = llemon.EmbedRequest(
            embedder=self.embedder,
            text=text,
            **provider_options,
        )
        response = await self.embedder.provider.embed(request)
        if save:
            self.history.append((request, response))
        return response

    async def transcribe(
        self,
        audio: FileArgument,
        save: bool = True,
        instructions: str | None = None,
        language: str | None = None,
        timestamps: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> TranscribeResponse:
        self._assert_not_finished()
        if not self.stt:
            raise Error(f"{self!r} has no STT associated with it")
        request = llemon.TranscribeRequest(
            stt=self.stt,
            audio=audio,
            instructions=instructions,
            language=language,
            timestamps=timestamps,
            timeout=timeout,
            **provider_options,
        )
        response = await self.stt.provider.transcribe(request)
        if save:
            self.history.append((request, response))
        return response

    async def synthesize(
        self,
        text: str,
        save: bool = True,
        voice: str | None = None,
        output_format: str | None = None,
        instructions: str | None = None,
        timestamps: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> SynthesizeResponse:
        self._assert_not_finished()
        if not self.tts:
            raise Error(f"{self!r} has no TTS associated with it")
        request = llemon.SynthesizeRequest(
            tts=self.tts,
            text=text,
            voice=voice,
            output_format=output_format,
            instructions=instructions,
            timestamps=timestamps,
            timeout=timeout,
            **provider_options,
        )
        response = await self.tts.provider.synthesize(request)
        if save:
            self.history.append((request, response))
        return response

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        embedder_model = unpacker.get("embedder", str, None)
        stt_model = unpacker.get("stt", str, None)
        tts_model = unpacker.get("tts", str, None)
        return cls(
            llm=refs.get_llm(unpacker.get("llm", str)),
            stt=refs.get_stt(stt_model) if stt_model else None,
            tts=refs.get_tts(tts_model) if tts_model else None,
            embedder=refs.get_embedder(embedder_model) if embedder_model else None,
            instructions=unpacker.get("instructions", str, None),
            context=unpacker.get("context", dict, None),
            render=llemon.Rendering.resolve(unpacker.get("render", (bool, str), None)),
            tools=[refs.get_tool(name) for name in unpacker.get("tools", list, [])],
            history=refs.get_history(unpacker.get("history", list, [])),
            include_messages=unpacker.get("include_messages", int, None),
            cache=unpacker.get("cache", bool, None),
        )

    def _dump(self, refs: DumpRefs) -> NS:
        refs.add_llm(self.llm)
        if self.stt:
            refs.add_stt(self.stt)
        if self.tts:
            refs.add_tts(self.tts)
        if self.embedder:
            refs.add_embedder(self.embedder)
        for request, response in self.history:
            refs.add_request(request)
            refs.add_response(response)
        for tool in self.tools:
            refs.add_tool(tool)
        return filtered_dict(
            llm=self.llm.model,
            stt=self.stt.model if self.stt else None,
            tts=self.tts.model if self.tts else None,
            embedder=self.embedder.model if self.embedder else None,
            instructions=self.instructions,
            context=self.context,
            render=self.rendering.bracket if self.rendering else False,
            tools=[tool.name for tool in self.tools],
            history=[request.id for request, _ in self.history],
            include_messages=self.include_messages,
            cache=self.cache,
        )

    def _assert_not_finished(self) -> None:
        if self.finished:
            raise Error(f"{self!r} has already finished")

    def _copy_history(self) -> list[tuple[Request, Response]]:
        history = []
        for request, response in self.history:
            if isinstance(request, GenerateRequest):
                request = copy.copy(request)
                files: list[File] = []
                for file in request.files:
                    file = copy.copy(file)
                    file.id = None
                    files.append(file)
                request.files = files
            history.append((request, response))
        return history

    def _resolve_context(self, context: NS | None) -> NS:
        if not context:
            return self.context
        return self.context | context

    def _resolve_history(self, include_messages: int | None) -> HistoryArgument:
        if include_messages is None:
            include_messages = self.include_messages
        if include_messages is None:
            return self.history[:]
        return self.history[-include_messages:]

    def _resolve_tools(self, tools: ToolsArgument) -> ToolsArgument:
        if not tools:
            return self.tools
        return [*self.tools, *tools]
