from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .genai.config import CONFIGS, Config
    from .genai.embedder.embedder import Embedder
    from .genai.embedder.embedder_property import EmbedderProperty
    from .genai.embedder.embedder_provider import EmbedderProvider
    from .genai.llm.llm import LLM
    from .genai.llm.llm_config import LLMConfig
    from .genai.llm.llm_property import LLMProperty
    from .genai.llm.llm_provider import LLMProvider
    from .genai.llm.tokenizers.llm_tokenizer import LLMToken, LLMTokenizer
    from .genai.provider import Provider
    from .genai.providers.anthropic import Anthropic
    from .genai.providers.deepinfra import DeepInfra
    from .genai.providers.gemini import Gemini
    from .genai.providers.ollama import Ollama
    from .genai.providers.openai import OpenAI
    from .genai.providers.openai_embedder import OpenAIEmbedder
    from .genai.providers.openai_llm import OpenAILLM
    from .genai.providers.openai_stt import OpenAISTT
    from .genai.stt.stt import STT
    from .genai.stt.stt_config import STTConfig
    from .genai.stt.stt_property import STTProperty
    from .genai.stt.stt_provider import STTProvider
    from .objects.call import Call
    from .objects.conversation import Conversation
    from .objects.file import File
    from .objects.protocol.classify import ClassifyRequest, ClassifyResponse
    from .objects.protocol.embed import EmbedRequest, EmbedResponse
    from .objects.protocol.generate import GenerateRequest, GenerateResponse
    from .objects.protocol.generate_object import GenerateObjectRequest, GenerateObjectResponse
    from .objects.protocol.generate_stream import GenerateStreamRequest, GenerateStreamResponse
    from .objects.protocol.request import Request, Response
    from .objects.protocol.transcribe import TranscribeRequest, TranscribeResponse
    from .objects.rendering import Rendering
    from .objects.serializeable import Serializeable
    from .objects.tool import Tool
    from .objects.toolbox import Toolbox
    from .tools.database import Database
    from .tools.directory import Directory
    from .types import Error, Warning
    from .utils import enable_logs


class __Importer:

    def __init__(self) -> None:
        self.sources: dict[str, str] = {}
        self.objects: dict[str, Any] = {}
        self.re = __import__("re")
        self.importlib = __import__("importlib")

    def __call__(self, name: str) -> Any:
        if not self.sources:
            self.load_sources()
        if name not in self.objects:
            if name not in self.sources:
                raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
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


__getattr__ = __Importer()


__all__ = [
    "Config",
    "CONFIGS",
    "Provider",
    "LLM",
    "LLMConfig",
    "LLMProperty",
    "LLMProvider",
    "LLMTokenizer",
    "LLMToken",
    "Embedder",
    "EmbedderProperty",
    "EmbedderProvider",
    "STT",
    "STTConfig",
    "STTProperty",
    "STTProvider",
    "OpenAILLM",
    "OpenAISTT",
    "OpenAIEmbedder",
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
    "TranscribeRequest",
    "TranscribeResponse",
    "File",
    "Call",
    "Tool",
    "Toolbox",
    "Rendering",
    "Serializeable",
    "Directory",
    "Database",
    "Error",
    "Warning",
    "enable_logs",
]
