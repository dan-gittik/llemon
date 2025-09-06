from __future__ import annotations

import llemon


class TTSConfig(llemon.Config):
    supports_timestamps: bool | None = None
    supports_streaming: bool | None = None
    supports_formats: list[str] | None = None
    cost_per_1m_characters: float | None = None
    cost_per_1m_tokens: float | None = None
