from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self, cast

from srlz import Serialization, SimpleType

import llemon
from llemon.types import NS, History
from llemon.utils import Unpacker, concat

if TYPE_CHECKING:
    from llemon import LLM, STT, TTS, Embedder, File, Request, Response, Tool, Toolbox

LLM_MODELS = "llm_models"
EMBEDDER_MODELS = "embedder_models"
STT_MODELS = "stt_models"
TTS_MODELS = "tts_models"
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
        self._add(LLM_MODELS, llm.model, llm, "model")

    def add_stt(self, stt: STT) -> None:
        self._add(STT_MODELS, stt.model, stt, "model")

    def add_tts(self, tts: TTS) -> None:
        self._add(TTS_MODELS, tts.model, tts, "model")

    def add_embedder(self, embedder: Embedder) -> None:
        self._add(EMBEDDER_MODELS, embedder.model, embedder, "model")

    def add_file(self, file: File) -> None:
        self._add(FILES, file.name, file, "name")

    def add_tool(self, tool: Tool | Toolbox) -> None:
        if isinstance(tool, llemon.Tool) and tool.toolbox:
            self._add(TOOLS, tool.toolbox.name, tool.toolbox, "name")
        else:
            self._add(TOOLS, tool.name, tool, "name")

    def add_request(self, request: Request) -> None:
        self._add(REQUESTS, request.id, request, "id")

    def add_response(self, response: Response) -> None:
        self._add(RESPONSES, response.request.id, response)

    def _add(self, tag: str, key: str, value: Serializeable, key_name: str | None = None) -> None:
        values: dict[str, NS] = self.refs.setdefault(tag, {})
        if key not in values:
            data = value._dump(self)
            if key_name:
                data.pop(key_name, None)
            values[key] = data


class LoadRefs:

    def __init__(self, unpacker: Unpacker) -> None:
        self.llms: dict[str, LLM] = {}
        for model, llm in unpacker.load_dict(LLM_MODELS, required=False):
            llm.data["model"] = model
            self.llms[model] = llemon.LLM._load(llm, self)
        self.stts: dict[str, STT] = {}
        for model, stt in unpacker.load_dict(STT_MODELS, required=False):
            stt.data["model"] = model
            self.stts[model] = llemon.STT._load(stt, self)
        self.tts: dict[str, TTS] = {}
        for model, tts in unpacker.load_dict(TTS_MODELS, required=False):
            tts.data["model"] = model
            self.tts[model] = llemon.TTS._load(tts, self)
        self.embedders: dict[str, Embedder] = {}
        for model, embedder in unpacker.load_dict(EMBEDDER_MODELS, required=False):
            embedder.data["model"] = model
            self.embedders[model] = llemon.Embedder._load(embedder, self)
        self.files: dict[str, File] = {}
        for name, file in unpacker.load_dict(FILES, required=False):
            file.data["name"] = name
            self.files[name] = llemon.File._load(file, self)
        self.tools: dict[str, Tool | Toolbox] = {}
        for name, tool in unpacker.load_dict(TOOLS, required=False):
            tool.data["name"] = name
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
            request.data["id"] = request_id
            self.requests[request_id] = llemon.Request._load(request, self)
            self.responses[request_id] = llemon.Response._load(responses[request_id], self)

    def get_llm(self, model: str) -> LLM:
        return self._get("LLM", model, self.llms)

    def get_stt(self, model: str) -> STT:
        return self._get("STT", model, self.stts)

    def get_tts(self, model: str) -> TTS:
        return self._get("TTS", model, self.tts)

    def get_embedder(self, model: str) -> Embedder:
        return self._get("embedder", model, self.embedders)

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

    def _get[T](self, label: str, key: str, data: dict[str, T]) -> T:
        if key not in data:
            raise ValueError(f"{label} {key!r} does not exist (available {label}s are {concat(data)})")
        return data[key]


@Serializeable.serialization.serializer("date", dt.date)
def serialize_date(date: dt.date) -> str:
    return date.isoformat()


@Serializeable.serialization.deserializer("date")
def deserialize_date(date: SimpleType) -> dt.date:
    date = cast(str, date)
    return dt.date.fromisoformat(date)
