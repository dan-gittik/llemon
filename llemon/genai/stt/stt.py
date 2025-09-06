from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

import llemon
from llemon.types import NS
from llemon.utils import filtered_dict

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
        instructions: str | None = None,
        language: str | None = None,
        timestamps: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> TranscribeResponse:
        request = llemon.TranscribeRequest(
            stt=self,
            audio=audio,
            instructions=instructions,
            language=language,
            timestamps=timestamps,
            timeout=timeout,
            **provider_options,
        )
        return await self.provider.transcribe(request)

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        provider = llemon.STTProvider.get_subclass(unpacker.get("provider", str))
        stt = provider.stt(model=unpacker.get("model", str), **unpacker.get("config", dict, {}))
        return cast(Self, stt)

    def _dump(self, refs: DumpRefs) -> NS:
        config = self.config._dump(refs)
        del config["model"]
        return filtered_dict(
            provider=self.provider.__class__.__name__,
            model=self.model,
            config=config,
        )


STTModel = llemon.Model[STT]
