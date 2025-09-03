from __future__ import annotations

import llemon


class EmbedderConfig(llemon.Config):
    category = llemon.CONFIGS["embedder"]

    cost_per_1m_tokens: float | None = None
