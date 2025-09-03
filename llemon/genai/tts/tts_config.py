from __future__ import annotations

import llemon


class TTSConfig(llemon.Config):
    category = llemon.CONFIGS["tts"]

    supports_timestamps: bool | None = None
    cost_per_1m_characters: float | None = None
    cost_per_1m_tokens: float | None = None
