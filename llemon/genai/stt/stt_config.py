from __future__ import annotations

import llemon


class STTConfig(llemon.Config):
    category = llemon.CONFIGS["stt"]

    supports_timestamps: bool | None = None
