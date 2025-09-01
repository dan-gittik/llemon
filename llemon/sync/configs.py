from __future__ import annotations

from importlib import resources

import yaml

from llemon.sync.types import NS

CONFIGS_PATH = resources.files("llemon.genai") / "configs.yaml"
CONFIGS: dict[str, dict[str, NS]] = yaml.safe_load(CONFIGS_PATH.read_text())
LLM_CONFIGS = CONFIGS["llm"]
