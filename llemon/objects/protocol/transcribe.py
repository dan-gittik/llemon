from __future__ import annotations

import warnings
from decimal import Decimal
from functools import cached_property
from typing import TYPE_CHECKING, Any

import llemon
from llemon.types import NS, FileArgument, Timestamps, Warning
from llemon.utils import Emoji, filtered_dict

if TYPE_CHECKING:
    from llemon import STT
    from llemon.objects.serializeable import DumpRefs, LoadRefs, Unpacker


class TranscribeRequest(llemon.Request):

    def __init__(
        self,
        *,
        stt: STT,
        audio: FileArgument,
        instructions: str | None = None,
        language: str | None = None,
        timestamps: bool | None = None,
        timeout: float | None = None,
        **provider_options: Any,
    ) -> None:
        super().__init__(self._overrides(stt.provider, provider_options))
        self.stt = stt
        self.audio = llemon.File.resolve(audio)
        self.instructions = instructions
        self.language = language
        self.timestamps = timestamps
        self.timeout = timeout

    def __str__(self) -> str:
        return f"{self.stt}.transcribe({self.audio!r})"

    def format(self, emoji: bool = True) -> str:
        embed = Emoji.TRANSCRIBE if emoji else "Transcribe: " + self.audio.name
        return f"{embed}{self.audio}"

    def check_supported(self) -> None:
        if self.timestamps and not self.stt.config.supports_timestamps:
            warnings.warn(f"{self.stt} doesn't support timestamps", Warning)
            self.timestamps = None

    @classmethod
    def _restore(cls, unpacker: Unpacker, refs: LoadRefs) -> tuple[NS, NS]:
        args = filtered_dict(
            stt=refs.get_stt(unpacker.get("stt", str)),
            audio=refs.get_file(unpacker.get("audio", str)),
            instructions=unpacker.get("instructions", str, None),
            language=unpacker.get("language", str, None),
            timestamps=unpacker.get("timestamps", bool, None),
            timeout=unpacker.get("timeout", float, None),
        )
        return args, {}

    def _dump(self, refs: DumpRefs) -> NS:
        data = super()._dump(refs)
        refs.add_stt(self.stt)
        refs.add_file(self.audio)
        data.update(
            filtered_dict(
                stt=self.stt.model,
                audio=self.audio.name,
            )
        )
        return data


class TranscribeResponse(llemon.Response):

    request: TranscribeRequest

    def __init__(self, request: TranscribeRequest) -> None:
        super().__init__(request)
        self.text: str | None = None
        self.timestamps: Timestamps | None = None
        self.input_tokens = 0
        self.duration = 0.0

    def __str__(self) -> str:
        return f"{self.request.stt}: {self.text!r}"

    @cached_property
    def cost(self) -> Decimal:
        if self.request.stt.config.cost_per_1m_input_tokens:
            return Decimal(self.input_tokens) * Decimal(self.request.stt.config.cost_per_1m_input_tokens) / 1_000_000
        return Decimal(self.duration) * Decimal(self.request.stt.config.cost_per_minute or 0) / 60.0

    def complete_transcription(self, text: str, timestamps: Timestamps | None = None) -> None:
        self.text = text
        self.timestamps = timestamps
        self.complete()

    def format(self, emoji: bool = True) -> str:
        embed = Emoji.TRANSCRIBE if emoji else "Transcribe: "
        return f"{embed}{self.text!r}"

    def _restore(self, unpacker: Unpacker, refs: LoadRefs) -> None:
        self.text = unpacker.get("text", str)
        self.timestamps = unpacker.get("timestamps", list, None)

    def _dump(self, refs: DumpRefs) -> NS:
        data = super()._dump(refs)
        data.update(
            filtered_dict(
                text=self.text,
                timestamps=self.timestamps,
            )
        )
        return data
