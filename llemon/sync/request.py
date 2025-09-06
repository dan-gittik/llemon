from __future__ import annotations

import datetime as dt
import json
import uuid
from functools import cached_property
from typing import TYPE_CHECKING, Any, Self

import llemon.sync as llemon
from llemon.sync.types import NS, Error
from llemon.utils import Superclass, filtered_dict, now

if TYPE_CHECKING:
    from llemon.sync import Provider
    from llemon.sync.serializeable import DumpRefs, LoadRefs, Unpacker


class Request(Superclass, llemon.Serializeable):

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        if overrides is None:
            overrides = {}
        self.id = str(uuid.uuid4())
        self.overrides = overrides
        self.state: NS = {}
        self.cleanup = True

    def __str__(self) -> str:
        return "request"

    def __repr__(self) -> str:
        return f"<{self}>"

    def error(self, message: str) -> RequestError:
        return RequestError(self, message)

    def format(self, emoji: bool = True) -> str:
        raise NotImplementedError()

    def check_supported(self) -> None:
        pass

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        request_class = cls.get_subclass(unpacker.get("type", str))
        args, attrs = request_class._restore(unpacker, refs)
        request = request_class(**filtered_dict(**args))
        for name, value in attrs.items():
            setattr(request, name, value)
        return request

    @classmethod
    def _restore(cls, unpacker: Unpacker, refs: LoadRefs) -> tuple[NS, NS]:
        return unpacker.get("overrides", dict, {}), {}

    def _dump(self, refs: DumpRefs) -> NS:
        return dict(
            type=self.__class__.__name__,
            overrides=self.overrides,
        )

    def _overrides(self, provider: Provider, provider_options: NS) -> NS:
        prefix = f"{provider.__class__.__name__.lower()}_"
        overrides: NS = {}
        for key, value in provider_options.items():
            if key.startswith(prefix):
                overrides[key.removeprefix(prefix)] = value
        return overrides


class Response(Superclass, llemon.Serializeable):

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

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        response_class = cls.get_subclass(unpacker.get("type", str))
        request = refs.get_request(unpacker.get("request", str))
        response = response_class(request)
        response.started = dt.datetime.fromisoformat(unpacker.get("started", str))
        response.ended = dt.datetime.fromisoformat(unpacker.get("ended", str))
        response_class._restore(response, unpacker, refs)
        return response

    def _restore(self, unpacker: Unpacker, refs: LoadRefs) -> None:
        pass

    def _dump(self, refs: DumpRefs) -> NS:
        if not self.ended:
            raise self._incomplete_request()
        refs.add_request(self.request)
        return filtered_dict(
            type=self.__class__.__name__,
            request=self.request.id,
            started=self.started.isoformat(),
            ended=self.ended.isoformat(),
        )

    def _incomplete_request(self) -> Error:
        return Error(f"{self.request} hasn't completed yet")


class RequestError(Error):

    def __init__(self, request: Request, message: str) -> None:
        super().__init__(message)
        self.request = request

    def __str__(self) -> str:
        return f"{super().__str__()}:\n{json.dumps(self.request.dump(), indent=2)}"
