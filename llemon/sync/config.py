from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING, Any, ClassVar, Self

import yaml
from pydantic import BaseModel

import llemon.sync as llemon
from llemon.sync.types import NS
from llemon.utils import filtered_dict

if TYPE_CHECKING:
    from llemon.sync.serializeable import DumpRefs, LoadRefs, Unpacker


class Config(BaseModel, llemon.Serializeable):
    model: str
    category: ClassVar[dict[str, NS]] = {}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**filtered_dict(**kwargs))
        if self.model in self.category:
            set_fields = self.model_dump(exclude_unset=True)
            for key, value in self.category[self.model].items():
                if key not in set_fields:
                    setattr(self, key, value)

    @classmethod
    def _load(cls, unpacker: Unpacker, refs: LoadRefs) -> Self:
        return cls(**unpacker.data)

    def _dump(self, refs: DumpRefs) -> NS:
        if self.model not in self.category:
            return self.model_dump()
        data = self.model_dump()
        for key, value in self.category[self.model].items():
            if data[key] == value:
                del data[key]
        return data


CONFIGS_PATH = resources.files("llemon.genai") / "configs.yaml"
CONFIGS: dict[str, dict[str, dict[str, NS]]] = yaml.safe_load(CONFIGS_PATH.read_text())
