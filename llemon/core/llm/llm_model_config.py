import datetime as dt
import pathlib

import yaml
from pydantic import BaseModel


class LLMModelConfig(BaseModel):
    name: str
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

    def load_defaults(self) -> None:
        if self.name not in LLM_MODEL_CONFIGS:
            return
        set_fields = self.model_dump(exclude_none=True)
        for key, value in LLM_MODEL_CONFIGS[self.name].model_dump().items():
            if key not in set_fields:
                setattr(self, key, value)


LLM_MODEL_CONFIGS: dict[str, LLMModelConfig] = {}

CONFIGS_PATH = pathlib.Path(__file__).parent / "configs.yaml"
for name, config in yaml.safe_load(CONFIGS_PATH.read_text()).items():
    if name.startswith("_"):
        continue
    LLM_MODEL_CONFIGS[name] = LLMModelConfig(name=name, **config)
