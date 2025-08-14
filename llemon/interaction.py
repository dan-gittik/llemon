from __future__ import annotations

import datetime as dt
from functools import cached_property
import logging
import re
from typing import Iterator, Self, overload

from .file import File
from .tool import Call
from .utils import USER, ASSISTANT, FILE, TOOL, now
from .types import InteractionArgument, HistoryArgument, Messages, UserMessage, AssistantMessage

SPACES = re.compile(r"\s+")

log = logging.getLogger(__name__)


class Interaction:

    def __init__(
        self,
        user: str,
        files: list[File] | None = None,
        assistant: str | None = None,
        calls: list[Call] | None = None,
    ) -> None:
        if files is None:
            files = []
        if calls is None:
            calls = []
        self.user = user
        self.files = files
        self.started = now()
        self.calls = calls
        self._assistant = assistant
        self._ended: dt.datetime | None = None
        self._ttft: float | None = None
        if self._assistant:
            self._ended = self.started
            self._ttft = 0.0
    
    def __str__(self) -> str:
        return f"interaction ({self.format()})"
    
    def __repr__(self) -> str:
        return f"<{self}>"
    
    @classmethod
    def resolve(cls, interaction: InteractionArgument) -> Interaction:
        if isinstance(interaction, Interaction):
            return interaction
        user, *calls, assistant = interaction
        files = [File.from_dict(dict_) for dict_ in user.get("files", [])]
        return cls(user["content"], files, assistant["content"], [Call.resolve(call) for call in calls])
    
    @cached_property
    def assistant(self) -> str:
        if self._assistant is None:
            raise self._didnt_complete()
        return self._assistant
    
    @cached_property
    def ended(self) -> dt.datetime:
        if self._ended is None:
            raise self._didnt_complete()
        return self._ended
    
    @cached_property
    def ttft(self) -> float:
        if self._ttft is None:
            raise self._didnt_complete()
        return self._ttft
    
    @cached_property
    def duration(self) -> float:
        if not self._ended:
            return 0.0
        return (self.ended - self.started).total_seconds()

    def end(self, assistant: str, calls: list[Call] | None = None, ttft: float | None = None) -> None:
        if calls:
            self.calls.extend(calls)
        self._assistant = assistant
        self._ended = now()
        self._ttft = ttft or self.duration
    
    def format(self) -> str:
        messages = [f"User: {self._condense(self.user)}"]
        if self._assistant:
            messages.append(f"Assistant: {self._condense(self.assistant)}")
        return " | ".join(messages)
    
    def to_messages(self) -> Messages:
        messages: Messages = []
        user_message = UserMessage(content=self.user)
        if self.files:
            for file in self.files:
                user_message.files.append(file.to_dict())
        messages.append(user_message)
        for call in self.calls:
            messages.append(call.to_message())
        messages.append(AssistantMessage(content=self.assistant))
        return messages
    
    def _didnt_complete(self) -> RuntimeError:
        return RuntimeError("interaction didn't completed yet")
    
    def _condense(self, text: str) -> str:
        return SPACES.sub(" ", text).strip()


class History:

    def __init__(self, interactions: list[Interaction] | None = None) -> None:
        if interactions is None:
            interactions = []
        self.interactions: list[Interaction] = interactions
    
    def __str__(self) -> str:
        return f"history ({self.format()})"
    
    def __repr__(self) -> str:
        return f"<{self}>"

    def __bool__(self) -> bool:
        return bool(self.interactions)
    
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __iter__(self) -> Iterator[Interaction]:
        yield from self.interactions

    @overload
    def __getitem__(self, index: int) -> Interaction: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> Interaction | Self:
        if isinstance(index, slice):
            return type(self)(self.interactions[index])
        return self.interactions[index]
    
    @classmethod
    def resolve(cls, history: HistoryArgument) -> History:
        if history is None:
            return cls()
        if isinstance(history, History):
            return history
        messages: list[dict[str, str]] = []
        interactions: list[Interaction] = []
        for message in history:
            match message:
                case {"role": "user" | "call"}:
                    messages.append(message)
                case {"role": "assistant"}:
                    messages.append(message)
                    interactions.append(Interaction.resolve(messages))
                    messages.clear()
                case _:
                    raise ValueError(
                        f"invalid message: {message} (expected message with role 'user', 'call' or 'assistant')"
                    )
        return cls(interactions)
    
    def append(self, interaction: Interaction) -> None:
        self.interactions.append(interaction)
    
    def format(self) -> str:
        return " | ".join(interaction.format() for interaction in self.interactions)
    
    def log(self) -> None:
        extra = {"markup": True, "highlighter": None}
        log.debug("[bold underline]history[/]", extra=extra)
        for n, interaction in enumerate(self.interactions, 1):
            timestamp = f"{interaction.started:%H:%M:%S}-{interaction.ended:%H:%M:%S} ({interaction.duration:.2f}s)"
            log.debug(f"[bold yellow]interaction #{n}[/] ({timestamp})", extra=extra)
            log.debug(f"{USER}{interaction.user}", extra=extra)
            for file in interaction.files:
                log.debug(f"{FILE}{file.name}", extra=extra)
            for call in interaction.calls:
                result = f"[red]{call.result['error']}[/]" if "error" in call.result else call.result["return_value"]
                log.debug(f"{TOOL}{call.signature} -> {result}", extra=extra)
            log.debug(f"{ASSISTANT}{interaction.assistant}", extra=extra)
    
    def to_messages(self) -> Messages:
        messages: Messages = []
        for interaction in self.interactions:
            messages.extend(interaction.to_messages())
        return messages