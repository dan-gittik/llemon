from __future__ import annotations

import datetime as dt

import llemon


class LLMConfig(llemon.Config):
    tokenizer: str | None = None
    knowledge_cutoff: dt.date | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    unsupported_parameters: list[str] | None = None
    supports_streaming: bool | None = None
    supports_structured_output: bool | None = None
    supports_json: bool | None = None
    supports_tools: bool | None = None
    supports_logit_biasing: bool | None = None
    accepts_files: list[str] | None = None
    cost_per_1m_input_tokens: float | None = None
    cost_per_1m_cache_tokens: float | None = None
    cost_per_1m_output_tokens: float | None = None

    @property
    def supports_objects(self) -> bool:
        return bool(self.supports_structured_output or self.supports_json)
