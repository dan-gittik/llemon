from __future__ import annotations

import llemon.sync as llemon


class STTConfig(llemon.Config):
    category = llemon.CONFIGS["stt"]

    supports_timestamps: bool | None = None
    cost_per_1m_input_tokens: float | None = None
    cost_per_minute: float | None = None
