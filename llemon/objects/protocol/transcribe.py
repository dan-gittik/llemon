from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

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
        prompt: str | None = None,
        language: str | None = None,
        timestamps: bool | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__()
        self.stt = stt
        self.audio = llemon.File.resolve(audio)
        self.prompt = prompt
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
            prompt=unpacker.get("prompt", str, None),
            language=unpacker.get("language", str, None),
            timestamps=unpacker.get("timestamps", bool, None),
            timeout=unpacker.get("timeout", float, None),
        )
        return args, {}

    def _dump(self, refs: DumpRefs) -> NS:
        refs.add_stt(self.stt)
        refs.add_file(self.audio)
        data = filtered_dict(
            stt=self.stt.model,
            audio=self.audio.name,
        )
        data.update(super()._dump(refs))
        return data


class TranscribeResponse(llemon.Response):

    request: TranscribeRequest

    def __init__(self, request: TranscribeRequest) -> None:
        super().__init__(request)
        self.text: str | None = None
        self.timestamps: Timestamps | None = None

    def __str__(self) -> str:
        return f"{self.request.stt}: {self.text!r}"

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
        data = filtered_dict(
            text=self.text,
            timestamps=self.timestamps,
        )
        data.update(super()._dump(refs))
        return data
