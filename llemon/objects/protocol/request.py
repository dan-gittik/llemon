from __future__ import annotations
import datetime as dt
from functools import cached_property

from llemon.types import NS, Error, History
from llemon.utils import Superclass, now


class Request(Superclass):

    def __init__(self, history: History | None = None) -> None:
        if history is None:
            history = []
        self.history = history
        self.id: str | None = None

    def __str__(self) -> str:
        return "request"

    def __repr__(self) -> str:
        return f"<{self}>"

    @classmethod
    def load(cls, data: NS) -> Request:
        from llemon.objects.serialization import load

        return load(cls, data["request"], data)

    def dump(self) -> NS:
        from llemon.objects.serialization import dump

        data, state = dump(self)
        return dict(
            request=data,
            **state,
        )

    def error(self, message: str) -> RequestError:
        return RequestError(self, message)

    def format(self, emoji: bool = True) -> str:
        raise NotImplementedError()

    def check_supported(self) -> None:
        pass


class Response(Superclass):

    def __init__(self, request: Request) -> None:
        self.request = request
        self.started = now()
        self.ended: dt.datetime | None = None

    def __str__(self) -> str:
        return "response"

    def __repr__(self) -> str:
        return f"<{self}>"

    @classmethod
    def load(cls, data: NS) -> Response:
        from llemon.objects.serialization import load

        request = Request.load(data["request"])
        return load(cls, data["response"], data, request=request)

    @cached_property
    def duration(self) -> float:
        if not self.ended:
            raise self._incomplete_request()
        return (self.ended - self.started).total_seconds()

    def dump(self) -> NS:
        from llemon.objects.serialization import dump

        request = self.request.dump()
        data, state = dump(self)
        return dict(
            request=request,
            response=data,
            **state,
        )

    def complete(self) -> None:
        self.ended = now()

    def format(self, emoji: bool = True) -> str:
        raise NotImplementedError()

    def _incomplete_request(self) -> Error:
        return Error(f"{self.request} hasn't completed yet")


class RequestError(Error):

    def __init__(self, request: Request, message: str) -> None:
        super().__init__(message)
        self.request = request
