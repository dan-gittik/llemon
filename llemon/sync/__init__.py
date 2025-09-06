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
    from .embedder import Embedder, EmbedderModel
    from .embedder_config import EmbedderConfig
    from .embedder_provider import EmbedderProvider
    from .file import File
    from .gemini import Gemini
    from .generate import GenerateRequest, GenerateResponse, GenerateStreamResponse
    from .generate_object import GenerateObjectRequest, GenerateObjectResponse
    from .llm import LLM, LLMModel
    from .llm_config import LLMConfig
    from .llm_provider import LLMProvider
    from .llm_tokenizer import LLMToken, LLMTokenizer
    from .model import Model
    from .ollama import Ollama
    from .openai import OpenAI
    from .openai_embedder import OpenAIEmbedder
    from .openai_llm import OpenAILLM
    from .openai_stt import OpenAISTT
    from .openai_tts import OpenAITTS
    from .provider import Provider
    from .rendering import Rendering
    from .request import Request, Response
    from .serializeable import Serializeable
    from .stt import STT, STTModel
    from .stt_config import STTConfig
    from .stt_provider import STTProvider
    from .synthesize import SynthesizeRequest, SynthesizeResponse
    from .tool import Tool
    from .toolbox import Toolbox
    from .transcribe import TranscribeRequest, TranscribeResponse
    from .tts import TTS, TTSModel
    from .tts_config import TTSConfig
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
    "Model",
    "LLM",
    "LLMModel",
    "LLMConfig",
    "LLMProperty",
    "LLMProvider",
    "LLMTokenizer",
    "LLMToken",
    "STT",
    "STTModel",
    "STTConfig",
    "STTProperty",
    "STTProvider",
    "TTS",
    "TTSModel",
    "TTSConfig",
    "TTSProperty",
    "TTSProvider",
    "Embedder",
    "EmbedderModel",
    "EmbedderProperty",
    "EmbedderProvider",
    "EmbedderConfig",
    "OpenAILLM",
    "OpenAISTT",
    "OpenAITTS",
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
    "GenerateStreamResponse",
    "GenerateObjectRequest",
    "GenerateObjectResponse",
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
