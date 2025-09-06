from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

import llemon
from llemon.types import NS
from llemon.utils import filtered_dict

if TYPE_CHECKING:
    from llemon import SynthesizeResponse, TTSConfig, TTSProvider
    from llemon.objects.serializeable import DumpRefs, LoadRefs, Unpacker


class TTS(llemon.Serializeable):

    def __init__(self, provider: TTSProvider, model: str, config: TTSConfig) -> None:
        self.provider = provider
        self.model = model
        self.config = config

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"

    def __repr__(self) -> str:
        return f"<{self}>"

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        output_format: str | None = None,
        instructions: str | None = None,
        timestamps: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> SynthesizeResponse:
        request = llemon.SynthesizeRequest(
            tts=self,
            text=text,
            voice=voice,
            output_format=output_format,
            instructions=instructions,
            timestamps=timestamps,
            timeout=timeout,
            **provider_options,
        )
        return await self.provider.synthesize(request)

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        provider = llemon.TTSProvider.get_subclass(unpacker.get("provider", str))
        tts = provider.tts(model=unpacker.get("model", str), **unpacker.get("config", dict, {}))
        return cast(Self, tts)

    def _dump(self, refs: DumpRefs) -> NS:
        config = self.config._dump(refs)
        del config["model"]
        return filtered_dict(
            provider=self.provider.__class__.__name__,
            model=self.model,
            config=config,
        )


TTSModel = llemon.Model[TTS]
