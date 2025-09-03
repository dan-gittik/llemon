from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, AsyncIterator

import llemon
from llemon.types import NS, Error, FilesArgument, HistoryArgument, RenderArgument, ToolsArgument
from llemon.utils import filtered_dict, now

if TYPE_CHECKING:
    from llemon import LLM
    from llemon.objects.serializeable import DumpRefs, LoadRefs, Unpacker


class GenerateStreamRequest(llemon.GenerateRequest):

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
        super().__init__(
            llm=llm,
            instructions=instructions,
            user_input=user_input,
            context=context,
            render=render,
            history=history,
            files=files,
            tools=tools,
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
            cache=cache,
            timeout=timeout,
            **provider_options,
        )

    def __str__(self) -> str:
        return f"{self.llm}.generate_stream({self.user_input!r})"

    def check_supported(self) -> None:
        super().check_supported()
        if not self.llm.config.supports_streaming:
            raise Error(f"{self.llm} doesn't support streaming")


class GenerateStreamResponse(llemon.GenerateResponse):

    request: GenerateStreamRequest

    def __init__(self, request: GenerateStreamRequest) -> None:
        super().__init__(request)
        self.stream: AsyncIterator[str] | None = None
        self._chunks: list[str] = []
        self._ttft: float | None = None

    def __str__(self) -> str:
        end = "..." if not self.ended else ""
        return f"{self.request.llm}: {'|'.join(self._chunks)}{end}"

    async def __aiter__(self) -> AsyncIterator[StreamDelta]:
        if self.stream is None:
            raise self._incomplete_request()
        async for chunk in self.stream:
            if self._ttft is None:
                self._ttft = (now() - self.started).total_seconds()
            self._chunks.append(chunk)
            yield StreamDelta(text=chunk)

    @cached_property
    def ttft(self) -> float:
        if not self.ended:
            raise self._incomplete_request()
        return self._ttft or 0.0

    def complete_stream(self) -> None:
        self.complete_text("".join(self._chunks))

    def _restore(self, unpacker: Unpacker, refs: LoadRefs) -> None:
        super()._restore(unpacker, refs)
        self._chunks = unpacker.get("chunks", list)
        self._ttft = unpacker.get("ttft", float)

    def _dump(self, refs: DumpRefs) -> NS:
        data = super()._dump(refs)
        data.update(
            filtered_dict(
                text=self.text,
                chunks=self._chunks,
                ttft=self.ttft,
            )
        )
        return data


class StreamDelta:

    def __init__(self, text: str) -> None:
        self.text = text
