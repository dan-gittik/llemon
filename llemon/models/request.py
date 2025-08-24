from __future__ import annotations

import copy
import datetime as dt
from functools import cached_property
from typing import Self

from llemon.errors import InProgressError
from llemon.types import NS, History
from llemon.utils.now import now


class Request:

    def __init__(self, history: History | None = None) -> None:
        if history is None:
            history = []
        self.history = history

    def __str__(self) -> str:
        return "request"

    def __repr__(self) -> str:
        return f"<{self}>"

    def check_supported(self) -> None:
        pass

    def format(self, emoji: bool = True) -> str:
        raise NotImplementedError()

    def _copy(self) -> Self:
        return copy.copy(self)


class Response:

    def __init__(self, request: Request) -> None:
        self.request = request
        self.started = now()
        self.ended: dt.datetime | None = None

    def __str__(self) -> str:
        return "response"

    def __repr__(self) -> str:
        return f"<{self}>"

    @cached_property
    def duration(self) -> float:
        if not self.ended:
            raise self._incomplete_request()
        return (self.ended - self.started).total_seconds()

    def complete(self) -> None:
        self.ended = now()

    def format(self, emoji: bool = True) -> str:
        raise NotImplementedError()

    def _copy(self) -> Self:
        return copy.copy(self)

    def _incomplete_request(self) -> InProgressError:
        return InProgressError(f"{self.request} hasn't completed yet")
