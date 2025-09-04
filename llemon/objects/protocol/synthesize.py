from __future__ import annotations

import warnings
from decimal import Decimal
from functools import cached_property
from typing import TYPE_CHECKING, Any

import llemon
from llemon.types import NS, Timestamps, Warning
from llemon.utils import Emoji, filtered_dict, get_extension, text_to_name

if TYPE_CHECKING:
    from llemon import TTS, File
    from llemon.objects.serializeable import DumpRefs, LoadRefs, Unpacker


class SynthesizeRequest(llemon.Request):

    def __init__(
        self,
        *,
        tts: TTS,
        text: str,
        voice: str | None = None,
        output_format: str | None = None,
        instructions: str | None = None,
        timestamps: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> None:
        super().__init__(self._overrides(tts.provider, provider_options))
        self.tts = tts
        self.text = text
        self.voice = voice
        self.output_format = output_format
        self.timestamps = timestamps
        self.instructions = instructions
        self.timeout = timeout

    def __str__(self) -> str:
        return f"{self.tts}.synthesize({self.text!r})"

    def format(self, emoji: bool = True) -> str:
        synthesize = Emoji.SYNTHESIZE if emoji else "Synthesize: "
        return f"{synthesize}{self.text}"

    def check_supported(self) -> None:
        if self.timestamps and not self.tts.config.supports_timestamps:
            warnings.warn(f"{self.tts} doesn't support timestamps", Warning)
            self.timestamps = None

    @classmethod
    def _restore(cls, unpacker: Unpacker, refs: LoadRefs) -> tuple[NS, NS]:
        args = filtered_dict(
            tts=refs.get_tts(unpacker.get("tts", str)),
            text=unpacker.get("text", str),
            voice=unpacker.get("voice", str, None),
            output_format=unpacker.get("output_format", str, None),
            instructions=unpacker.get("instructions", str, None),
            timestamps=unpacker.get("timestamps", bool, None),
            timeout=unpacker.get("timeout", float, None),
        )
        return args, {}

    def _dump(self, refs: DumpRefs) -> NS:
        data = super()._dump(refs)
        refs.add_tts(self.tts)
        data.update(
            filtered_dict(
                tts=self.tts.model,
                text=self.text,
                voice=self.voice,
                output_format=self.output_format,
                instructions=self.instructions,
                timestamps=self.timestamps,
                timeout=self.timeout,
            )
        )
        return data


class SynthesizeResponse(llemon.Response):

    request: SynthesizeRequest

    def __init__(self, request: SynthesizeRequest) -> None:
        super().__init__(request)
        self.audio: File | None = None
        self.timestamps: Timestamps | None = None
        self.output_tokens = 0

    def __str__(self) -> str:
        return f"{self.request.tts}: {self.audio}"

    @cached_property
    def cost(self) -> Decimal:
        if self.request.tts.config.cost_per_1m_characters:
            return Decimal(len(self.request.text)) * Decimal(self.request.tts.config.cost_per_1m_characters) / 1_000_000
        return Decimal(self.output_tokens) * Decimal(self.request.tts.config.cost_per_1m_tokens or 0) / 1_000_000

    def complete_synthesis(self, data: bytes, mimetype: str, timestamps: Timestamps | None = None) -> None:
        name = text_to_name(self.request.text, max_length=100, default="synthesis") + get_extension(mimetype)
        self.audio = llemon.File.from_data(data, mimetype, name)
        self.timestamps = timestamps
        self.complete()

    def format(self, emoji: bool = True) -> str:
        synthesize = Emoji.SYNTHESIZE if emoji else "Synthesis: "
        return f"{synthesize}{self.audio}"

    def _restore(self, unpacker: Unpacker, refs: LoadRefs) -> None:
        self.audio = refs.get_file(unpacker.get("audio", str))
        self.timestamps = unpacker.get("timestamps", list, None)

    def _dump(self, refs: DumpRefs) -> NS:
        data = super()._dump(refs)
        assert self.audio is not None
        refs.add_file(self.audio)
        data.update(
            filtered_dict(
                audio=self.audio.name,
                timestamps=self.timestamps,
            )
        )
        return data
