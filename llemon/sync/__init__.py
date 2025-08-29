from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .configs import LLM_CONFIGS
    from .llm import LLM
    from .llm_config import LLMConfig
    from .llm_property import LLMProperty
    from .llm_provider import LLMProvider
    from .llm_tokenizer import LLMToken, LLMTokenizer
    from .provider import Provider
    from .anthropic import Anthropic
    from .deepinfra import DeepInfra
    from .gemini import Gemini
    from .ollama import Ollama
    from .openai import OpenAI
    from .conversation import Conversation
    from .file import File
    from .classify import ClassifyRequest, ClassifyResponse
    from .embed import EmbedRequest, EmbedResponse
    from .generate import GenerateRequest, GenerateResponse
    from .generate_object import GenerateObjectRequest, GenerateObjectResponse
    from .generate_stream import GenerateStreamRequest, GenerateStreamResponse
    from .request import Request, Response
    from .rendering import Rendering
    from .serialization import dump, load, serialization
    from .tool import Call, Tool, Toolbox
    from .database import Database
    from .directory import Directory
    from .types import Error, Warning
    from ..utils import enable_logs


class __Importer:

    def __init__(self) -> None:
        self.sources: dict[str, str] = {}
        self.objects: dict[str, Any] = {}
        self.re = __import__("re")
        self.importlib = __import__("importlib")

    def __getattr__(self, name: str) -> Any:
        return self.get(name)

    def get(self, name: str) -> Any:
        if not self.sources:
            self.load_sources()
        if name not in self.objects:
            if name not in self.sources:
                raise AttributeError(name)
            module = self.importlib.import_module(self.sources[name], __package__)
            self.objects[name] = getattr(module, name)
        return self.objects[name]

    def load_sources(self) -> None:
        for source, names in self.re.findall(r"from (.*?) import (.*)", open(__file__).read()):
            if not source.startswith("."):
                continue
            for name in names.split(","):
                name = name.strip()
                self.sources[name] = source


__importer = __Importer()


def __getattr__(name: str):
    try:
        return __importer.get(name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


__all__ = [
    "LLM_CONFIGS",
    "Provider",
    "LLM",
    "LLMConfig",
    "LLMProperty",
    "LLMProvider",
    "LLMTokenizer",
    "LLMToken",
    "OpenAI",
    "Anthropic",
    "Gemini",
    "DeepInfra",
    "Ollama",
    "Conversation",
    "Request",
    "Response",
    "GenerateRequest",
    "GenerateResponse",
    "GenerateObjectRequest",
    "GenerateObjectResponse",
    "GenerateStreamRequest",
    "GenerateStreamResponse",
    "ClassifyRequest",
    "ClassifyResponse",
    "EmbedRequest",
    "EmbedResponse",
    "File",
    "Call",
    "Tool",
    "Toolbox",
    "Rendering",
    "dump",
    "load",
    "serialization",
    "Directory",
    "Database",
    "Error",
    "Warning",
    "enable_logs",
]
