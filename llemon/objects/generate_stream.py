from __future__ import annotations

from functools import cached_property
from typing import AsyncIterator

from llemon.objects.generate import GenerateRequest, GenerateResponse
from llemon.types import Error
from llemon.utils import now


class GenerateStreamRequest(GenerateRequest):

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
