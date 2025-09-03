from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

import llemon.sync as llemon
from llemon.sync.types import NS
from llemon.utils import filtered_dict

if TYPE_CHECKING:
    from llemon.sync import FileArgument, STTConfig, STTProvider, TranscribeResponse
    from llemon.sync.serializeable import DumpRefs, LoadRefs, Unpacker


class STT(llemon.Serializeable):

    def __init__(self, provider: STTProvider, model: str, config: STTConfig) -> None:
        self.provider = provider
        self.model = model
        self.config = config

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"

    def __repr__(self) -> str:
        return f"<{self}>"

    def transcribe(
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
        return self.provider.transcribe(request)

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        provider = llemon.STTProvider.get_subclass(unpacker.get("provider", str))
        config = unpacker.get("config", dict)
        config.pop("model", None)
        stt = provider.stt(model=unpacker.get("model", str), **config)
        return cast(Self, stt)

    def _dump(self, refs: DumpRefs) -> NS:
        return filtered_dict(
            provider=self.provider.__class__.__name__,
            model=self.model,
            config=self.config._dump(refs),
        )
