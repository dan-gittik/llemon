from __future__ import annotations

import llemon.sync as llemon


class EmbedderConfig(llemon.Config):
    cost_per_1m_tokens: float | None = None
