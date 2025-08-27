from __future__ import annotations

from functools import cached_property
from typing import AsyncIterator

from llemon.objects.generate import GenerateRequest, GenerateResponse
from llemon.types import NS, Error, FilesArgument, History, RenderArgument, ToolsArgument
from llemon.utils import now


class GenerateStreamRequest(GenerateRequest):

    def __init__(
        self,
        *,
        model: LLMModel,
        history: History | None = None,
        instructions: str | None = None,
        user_input: str | None = None,
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
    ) -> None:
        super().__init__(
            model=model,
            history=history,
            instructions=instructions,
            user_input=user_input,
            context=context,
            render=render,
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
            top_k=top_k,
            stop=stop,
            prediction=prediction,
            return_incomplete_message=return_incomplete_message,
        )

    def check_supported(self) -> None:
        super().check_supported()
        if not self.model.config.supports_streaming:
            raise Error(f"{self.model} doesn't support streaming")


class GenerateStreamResponse(GenerateResponse):

    request: GenerateStreamRequest

    def __init__(self, request: GenerateStreamRequest) -> None:
        super().__init__(request)
        self.stream: AsyncIterator[str] | None = None
        self._chunks: list[str] = []
        self._ttft: float | None = None

    def __str__(self) -> str:
        end = "..." if not self.ended else ""
        return f"{self.request.model}: {''.join(self._chunks)}{end}"

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
        super().complete_text("".join(self._chunks))


class StreamDelta:

    def __init__(self, text: str) -> None:
        self.text = text


from llemon.genai.llm_model import LLMModel
