from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast

import llemon
from llemon.types import NS

if TYPE_CHECKING:
    from llemon import FileArgument, STTConfig, STTProvider, TranscribeResponse
    from llemon.objects.serializeable import DumpRefs, LoadRefs, Unpacker


class STT(llemon.Serializeable):

    def __init__(self, provider: STTProvider, model: str, config: STTConfig) -> None:
        self.provider = provider
        self.model = model
        self.config = config

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"

    def __repr__(self) -> str:
        return f"<{self}>"

    async def transcribe(
        self,
        audio: FileArgument,
        prompt: str | None = None,
        language: str | None = None,
        timestamps: bool | None = None,
        timeout: float | None = None,
    ) -> TranscribeResponse:
        request = llemon.TranscribeRequest(
            stt=self,
            audio=audio,
            prompt=prompt,
            language=language,
            timestamps=timestamps,
            timeout=timeout,
        )
        return await self.provider.transcribe(request)

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        provider = llemon.STTProvider.get_subclass(unpacker.get("provider", str))
        return cast(Self, provider.stt(model=unpacker.get("model", str)))

    def _dump(self, refs: DumpRefs) -> NS:
        return dict(
            provider=self.provider.__class__.__name__,
            model=self.model,
        )
