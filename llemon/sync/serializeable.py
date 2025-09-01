from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self, cast

from srlz import Serialization, SimpleType

import llemon.sync as llemon
from llemon.sync.types import NS, History
from llemon.utils import Unpacker, concat

if TYPE_CHECKING:
    from llemon.sync import LLM, STT, Embedder, File, Request, Response, Tool, Toolbox

LLMS = "llms"
EMBEDDERS = "embedders"
STTS = "stts"
REQUESTS = "requests"
RESPONSES = "responses"
FILES = "files"
TOOLS = "tools"


class Serializeable(ABC):

    serialization: ClassVar[Serialization] = Serialization()

    @classmethod
    def load(cls, data: NS) -> Self:
        unpacker = Unpacker(cls.serialization.deserialize(data))
        return cls._load(unpacker.load("data"), LoadRefs(unpacker.load("refs")))

    def dump(self) -> NS:
        refs = DumpRefs()
        data = self._dump(refs)
        return self.serialization.serialize(
            dict(
                data=data,
                refs=refs.refs,
            )
        )

    @classmethod
    @abstractmethod
    def _load(cls, data: Unpacker, refs: LoadRefs) -> Self:
        pass

    @abstractmethod
    def _dump(self, refs: DumpRefs) -> NS:
        pass


class DumpRefs:

    def __init__(self) -> None:
        self.refs: NS = {}

    def add_llm(self, llm: LLM) -> None:
        self._add(LLMS, llm.model, llm)

    def add_embedder(self, embedder: Embedder) -> None:
        self._add(EMBEDDERS, embedder.model, embedder)

    def add_stt(self, stt: STT) -> None:
        self._add(STTS, stt.model, stt)

    def add_file(self, file: File) -> None:
        self._add(FILES, file.name, file)

    def add_tool(self, tool: Tool | Toolbox) -> None:
        if isinstance(tool, llemon.Tool) and tool.toolbox:
            self._add(TOOLS, tool.toolbox.name, tool.toolbox)
        else:
            self._add(TOOLS, tool.name, tool)

    def add_request(self, request: Request) -> None:
        self._add(REQUESTS, request.id, request)

    def add_response(self, response: Response) -> None:
        self._add(RESPONSES, response.request.id, response)

    def _add(self, tag: str, key: str, value: Serializeable) -> None:
        values: dict[str, NS] = self.refs.setdefault(tag, {})
        if key not in values:
            values[key] = value._dump(self)


class LoadRefs:

    def __init__(self, unpacker: Unpacker) -> None:
        self.llms: dict[str, LLM] = {}
        for model, llm in unpacker.load_dict("llms", required=False):
            self.llms[model] = llemon.LLM._load(llm, self)
        self.embedders: dict[str, Embedder] = {}
        for model, embedder in unpacker.load_dict(EMBEDDERS, required=False):
            self.embedders[model] = llemon.Embedder._load(embedder, self)
        self.stts: dict[str, STT] = {}
        for model, stt in unpacker.load_dict(STTS, required=False):
            self.stts[model] = llemon.STT._load(stt, self)
        self.files: dict[str, File] = {}
        for name, file in unpacker.load_dict(FILES, required=False):
            self.files[name] = llemon.File._load(file, self)
        self.tools: dict[str, Tool | Toolbox] = {}
        for name, tool in unpacker.load_dict(TOOLS, required=False):
            if "type" in tool.data:
                toolbox = llemon.Toolbox._load(tool, self)
                self.tools[name] = toolbox
                for tool_ in toolbox.tools:
                    self.tools[tool_.name] = tool_
            else:
                self.tools[name] = llemon.Tool._load(tool, self)
        self.requests: dict[str, Request] = {}
        self.responses: dict[str, Response] = {}
        responses = {request_id: response for request_id, response in unpacker.load_dict(RESPONSES)}
        for request_id, request in unpacker.load_dict(REQUESTS, required=False):
            self.requests[request_id] = llemon.Request._load(request, self)
            self.responses[request_id] = llemon.Response._load(responses[request_id], self)

    def get_llm(self, model: str) -> LLM:
        return self._get("LLM", model, self.llms)

    def get_embedder(self, model: str) -> Embedder:
        return self._get("embedder", model, self.embedders)

    def get_stt(self, model: str) -> STT:
        return self._get("stt", model, self.stts)

    def get_file(self, name: str) -> File:
        return self._get("file", name, self.files)

    def get_tool(self, name: str) -> Tool | Toolbox:
        return self._get("tool", name, self.tools)

    def get_request(self, request_id: str) -> Request:
        return self._get("request", request_id, self.requests)

    def get_response(self, request_id: str) -> Response:
        return self._get("response", request_id, self.responses)

    def get_history(self, request_ids: list[str]) -> History:
        history: History = []
        for request_id in request_ids:
            history.append((self.get_request(request_id), self.get_response(request_id)))
        return history

    def _get[T](self, label: str, name: str, data: dict[str, T]) -> T:
        if name not in data:
            raise ValueError(f"{label} {name!r} does not exist (available {label}s are {concat(data)})")
        return data[name]


@Serializeable.serialization.serializer("date", dt.date)
def serialize_date(date: dt.date) -> str:
    return date.isoformat()


@Serializeable.serialization.deserializer("date")
def deserialize_date(date: SimpleType) -> dt.date:
    date = cast(str, date)
    return dt.date.fromisoformat(date)
