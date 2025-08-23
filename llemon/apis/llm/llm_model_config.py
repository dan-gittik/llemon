import datetime as dt

from pydantic import BaseModel

from llemon.types import NS


class LLMModelConfig(BaseModel):
    knowledge_cutoff: dt.date | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_variants: bool | None = None
    supports_streaming: bool | None = None
    supports_structured_output: bool | None = None
    supports_json: bool | None = None
    supports_tools: bool | None = None
    supports_logit_biasing: bool | None = None
    accepts_files: list[str] | None = None
    cost_per_1m_input_tokens: float | None = None
    cost_per_1m_output_tokens: float | None = None

    def load_defaults(self, name: str) -> None:
        if name not in LLM_MODEL_CONFIGS:
            return
        set_fields = self.model_dump(exclude_none=True)
        for key, value in LLM_MODEL_CONFIGS[name].model_dump().items():
            if key not in set_fields:
                setattr(self, key, value)

    def dump(self, name: str) -> NS:
        if name not in LLM_MODEL_CONFIGS:
            return self.model_dump()
        data = self.model_dump()
        for key, value in LLM_MODEL_CONFIGS[name].model_dump().items():
            if data[key] == value:
                del data[key]
        return data


JPG = "image/jpeg"
PNG = "image/png"
GIF = "image/gif"
WEBP = "image/webp"
PDF = "application/pdf"
MP3 = "audio/mpeg"
WAV = "audio/wav"
FLAC = "audio/flac"
M4A = "audio/mp4"
MP4 = "video/mp4"
QUICKTIME = "video/quicktime"
WEBM = "video/webm"

LLM_MODEL_CONFIGS = {
    # OpenAI
    "gpt-5": LLMModelConfig(
        knowledge_cutoff=dt.date(2024, 10, 1),
        context_window=400_000,
        max_output_tokens=128_000,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=True,
        accepts_files=[JPG, PNG, GIF, PDF],
        cost_per_1m_input_tokens=1.25,
        cost_per_1m_output_tokens=10.0,
    ),
    "gpt-5-mini": LLMModelConfig(
        knowledge_cutoff=dt.date(2024, 5, 31),
        context_window=400_000,
        max_output_tokens=128_000,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=True,
        accepts_files=[JPG, PNG, GIF, PDF],
        cost_per_1m_input_tokens=0.25,
        cost_per_1m_output_tokens=2.0,
    ),
    "gpt-5-nano": LLMModelConfig(
        knowledge_cutoff=dt.date(2024, 5, 31),
        context_window=400_000,
        max_output_tokens=128_000,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=True,
        accepts_files=[JPG, PNG, GIF, PDF],
        cost_per_1m_input_tokens=0.05,
        cost_per_1m_output_tokens=0.4,
    ),
    "gpt-4.1": LLMModelConfig(
        knowledge_cutoff=dt.date(2024, 6, 1),
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=True,
        accepts_files=[JPG, PNG, GIF, PDF],
        cost_per_1m_input_tokens=2.0,
        cost_per_1m_output_tokens=8.0,
    ),
    "gpt-4.1-mini": LLMModelConfig(
        knowledge_cutoff=dt.date(2024, 6, 1),
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=True,
        accepts_files=[JPG, PNG, GIF, PDF],
        cost_per_1m_input_tokens=0.1,
        cost_per_1m_output_tokens=0.4,
    ),
    "gpt-4.1-nano": LLMModelConfig(
        knowledge_cutoff=dt.date(2024, 6, 1),
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=True,
        accepts_files=[JPG, PNG, GIF, PDF],
        cost_per_1m_input_tokens=0.1,
        cost_per_1m_output_tokens=0.4,
    ),
    "gpt-4o": LLMModelConfig(
        knowledge_cutoff=dt.date(2023, 10, 1),
        context_window=128_000,
        max_output_tokens=16_384,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=True,
        accepts_files=[JPG, PNG, GIF, PDF],
        cost_per_1m_input_tokens=2.5,
        cost_per_1m_output_tokens=10.0,
    ),
    "gpt-4o-mini": LLMModelConfig(
        knowledge_cutoff=dt.date(2023, 10, 1),
        context_window=128_000,
        max_output_tokens=16_384,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=True,
        accepts_files=[JPG, PNG, GIF, PDF],
        cost_per_1m_input_tokens=0.15,
        cost_per_1m_output_tokens=0.6,
    ),
    "gpt-4": LLMModelConfig(
        knowledge_cutoff=dt.date(2023, 12, 1),
        context_window=8192,
        max_output_tokens=8192,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=False,
        supports_json=False,
        supports_tools=False,
        supports_logit_biasing=True,
        accepts_files=[JPG, PNG, GIF, PDF],
        cost_per_1m_input_tokens=30.0,
        cost_per_1m_output_tokens=60.0,
    ),
    "gpt-4-turbo": LLMModelConfig(
        knowledge_cutoff=dt.date(2023, 12, 1),
        context_window=128_000,
        max_output_tokens=4096,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=False,
        supports_json=False,
        supports_tools=True,
        supports_logit_biasing=True,
        accepts_files=[JPG, PNG, GIF, PDF],
        cost_per_1m_input_tokens=10.0,
        cost_per_1m_output_tokens=30.0,
    ),
    "gpt-3.5-turbo": LLMModelConfig(
        knowledge_cutoff=dt.date(2021, 9, 1),
        context_window=16_385,
        max_output_tokens=4096,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=False,
        supports_json=False,
        supports_tools=False,
        supports_logit_biasing=True,
        accepts_files=[],
        cost_per_1m_input_tokens=0.5,
        cost_per_1m_output_tokens=1.5,
    ),
    # Anthropic
    "claude-opus-4-1": LLMModelConfig(
        knowledge_cutoff=dt.date(2025, 3, 1),
        context_window=200_000,
        max_output_tokens=32_000,
        supports_variants=False,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=False,
        accepts_files=[JPG, PNG, GIF, WEBP, PDF],
        cost_per_1m_input_tokens=15.0,
        cost_per_1m_output_tokens=75.0,
    ),
    "claude-opus-4-0": LLMModelConfig(
        knowledge_cutoff=dt.date(2025, 3, 1),
        context_window=200_000,
        max_output_tokens=32_000,
        supports_variants=False,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=False,
        accepts_files=[JPG, PNG, GIF, WEBP, PDF],
        cost_per_1m_input_tokens=15.0,
        cost_per_1m_output_tokens=75.0,
    ),
    "claude-sonnet-4-0": LLMModelConfig(
        knowledge_cutoff=dt.date(2025, 3, 1),
        context_window=200_000,
        max_output_tokens=64_000,
        supports_variants=False,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=False,
        accepts_files=[JPG, PNG, GIF, WEBP, PDF],
        cost_per_1m_input_tokens=3.0,
        cost_per_1m_output_tokens=15.0,
    ),
    "claude-3-7-sonnet-latest": LLMModelConfig(
        knowledge_cutoff=dt.date(2024, 11, 1),
        context_window=200_000,
        max_output_tokens=64_000,
        supports_variants=False,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=False,
        accepts_files=[JPG, PNG, GIF, WEBP, PDF],
        cost_per_1m_input_tokens=3.0,
        cost_per_1m_output_tokens=15.0,
    ),
    "claude-3-5-sonnet-latest": LLMModelConfig(
        knowledge_cutoff=dt.date(2024, 4, 1),
        context_window=200_000,
        max_output_tokens=8192,
        supports_variants=False,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=False,
        accepts_files=[JPG, PNG, GIF, WEBP, PDF],
        cost_per_1m_input_tokens=3.0,
        cost_per_1m_output_tokens=15.0,
    ),
    "claude-3-5-haiku-latest": LLMModelConfig(
        knowledge_cutoff=dt.date(2024, 7, 1),
        context_window=200_000,
        max_output_tokens=8192,
        supports_variants=False,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=False,
        accepts_files=[JPG, PNG, GIF, WEBP, PDF],
        cost_per_1m_input_tokens=0.8,
        cost_per_1m_output_tokens=4.0,
    ),
    "claude-3-haiku-20240307": LLMModelConfig(
        knowledge_cutoff=dt.date(2023, 8, 1),
        context_window=200_000,
        max_output_tokens=4096,
        supports_variants=False,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=False,
        accepts_files=[JPG, PNG, GIF, WEBP, PDF],
        cost_per_1m_input_tokens=0.25,
        cost_per_1m_output_tokens=1.25,
    ),
    # Gemini
    "gemini-2.5-pro": LLMModelConfig(
        knowledge_cutoff=dt.date(2025, 1, 1),
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=False,
        accepts_files=[JPG, PNG, PDF, MP3, WAV, M4A, MP4, QUICKTIME, WEBM],
        cost_per_1m_input_tokens=1.25,
        cost_per_1m_output_tokens=10.0,
    ),
    "gemini-2.5-flash": LLMModelConfig(
        knowledge_cutoff=dt.date(2025, 1, 1),
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=False,
        accepts_files=[JPG, PNG, PDF, MP3, WAV, M4A, MP4, QUICKTIME, WEBM],
        cost_per_1m_input_tokens=0.3,
        cost_per_1m_output_tokens=2.5,
    ),
    "gemini-2.5-flash-lite": LLMModelConfig(
        knowledge_cutoff=dt.date(2025, 1, 1),
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=False,
        accepts_files=[JPG, PNG, PDF, MP3, WAV, M4A, MP4, QUICKTIME, WEBM],
        cost_per_1m_input_tokens=0.1,
        cost_per_1m_output_tokens=0.4,
    ),
    "gemini-2.0-flash": LLMModelConfig(
        knowledge_cutoff=dt.date(2024, 8, 1),
        context_window=1_048_576,
        max_output_tokens=8192,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=False,
        accepts_files=[JPG, PNG, PDF, MP3, WAV, M4A, MP4, QUICKTIME, WEBM],
        cost_per_1m_input_tokens=0.1,
        cost_per_1m_output_tokens=0.4,
    ),
    "gemini-2.0-flash-lite": LLMModelConfig(
        knowledge_cutoff=dt.date(2024, 8, 1),
        context_window=1_048_576,
        max_output_tokens=8192,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=True,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=False,
        accepts_files=[JPG, PNG, PDF, MP3, WAV, M4A, MP4, QUICKTIME, WEBM],
        cost_per_1m_input_tokens=0.075,
        cost_per_1m_output_tokens=0.3,
    ),
    # DeepInfra
    "meta-llama/Meta-Llama-3.1-70B-Instruct": LLMModelConfig(
        knowledge_cutoff=dt.date(2023, 12, 1),
        context_window=128_000,
        max_output_tokens=2048,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=False,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=True,
        accepts_files=[],
        cost_per_1m_input_tokens=0.23,
        cost_per_1m_output_tokens=0.4,
    ),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": LLMModelConfig(
        knowledge_cutoff=dt.date(2023, 12, 1),
        context_window=128_000,
        max_output_tokens=2048,
        supports_variants=True,
        supports_streaming=True,
        supports_structured_output=False,
        supports_json=True,
        supports_tools=True,
        supports_logit_biasing=False,
        accepts_files=[],
        cost_per_1m_input_tokens=0.03,
        cost_per_1m_output_tokens=0.05,
    ),
}
