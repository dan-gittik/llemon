from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..utils import enable_logs
    from .anthropic import Anthropic
    from .call import Call
    from .classify import ClassifyRequest, ClassifyResponse
    from .config import CONFIGS, Config
    from .conversation import Conversation
    from .database import Database
    from .deepinfra import DeepInfra
    from .directory import Directory
    from .embed import EmbedRequest, EmbedResponse
    from .embedder import Embedder
    from .embedder_config import EmbedderConfig
    from .embedder_property import EmbedderProperty
    from .embedder_provider import EmbedderProvider
    from .file import File
    from .gemini import Gemini
    from .generate import GenerateRequest, GenerateResponse
    from .generate_object import GenerateObjectRequest, GenerateObjectResponse
    from .generate_stream import GenerateStreamRequest, GenerateStreamResponse
    from .llm import LLM
    from .llm_config import LLMConfig
    from .llm_property import LLMProperty
    from .llm_provider import LLMProvider
    from .llm_tokenizer import LLMToken, LLMTokenizer
    from .ollama import Ollama
    from .openai import OpenAI
    from .openai_embedder import OpenAIEmbedder
    from .openai_llm import OpenAILLM
    from .openai_stt import OpenAISTT
    from .provider import Provider
    from .rendering import Rendering
    from .request import Request, Response
    from .serializeable import Serializeable
    from .stt import STT
    from .stt_config import STTConfig
    from .stt_property import STTProperty
    from .stt_provider import STTProvider
    from .synthesize import SynthesizeRequest, SynthesizeResponse
    from .tool import Tool
    from .toolbox import Toolbox
    from .transcribe import TranscribeRequest, TranscribeResponse
    from .tts import TTS
    from .tts_config import TTSConfig
    from .tts_property import TTSProperty
    from .tts_provider import TTSProvider
    from .types import Error, Warning


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
    "STT",
    "STTConfig",
    "STTProperty",
    "STTProvider",
    "TTS",
    "TTSConfig",
    "TTSProperty",
    "TTSProvider",
    "Embedder",
    "EmbedderProperty",
    "EmbedderProvider",
    "EmbedderConfig",
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
    "SynthesizeRequest",
    "SynthesizeResponse",
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
