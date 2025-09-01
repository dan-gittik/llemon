from __future__ import annotations

import llemon.sync as llemon


class STTConfig(llemon.Config):
    category = llemon.CONFIGS["stt"]

    supports_timestamps: bool | None = None
