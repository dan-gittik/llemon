from __future__ import annotations

import os
from functools import cached_property
from typing import TYPE_CHECKING, Sequence, cast

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

import llemon

if TYPE_CHECKING:
    from llemon import LLM, LLMToken

TOKENIZERS: dict[str, PreTrainedTokenizerFast] = {}


class HFTokenizer(llemon.LLMTokenizer):

    def __init__(self, llm: LLM) -> None:
        super().__init__(llm)
        if llm.model not in TOKENIZERS:
            TOKENIZERS[llm.model] = AutoTokenizer.from_pretrained(llm.model, use_fast=False)
        self._tokenizer = TOKENIZERS[llm.model]

    async def parse(self, text: str) -> Sequence[LLMToken]:
        output = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens: list[HFToken] = []
        for id, (start, end) in zip(output["input_ids"], output["offset_mapping"]):
            token = HFToken(id, start, self._tokenizer)
            tokens.append(token)
        return tokens

    async def encode(self, *texts: str) -> list[int]:
        output = self._tokenizer(texts, add_special_tokens=False)
        return output["input_ids"][0]

    async def decode(self, *ids: int) -> str:
        tokens = cast(list[str], self._tokenizer.convert_ids_to_tokens(list(ids)))
        return self._tokenizer.convert_tokens_to_string(tokens)

    def _count(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])


class HFToken(llemon.LLMToken):

    def __init__(self, id: int, offset: int, tokenizer: PreTrainedTokenizerFast) -> None:
        self.id = id
        self._offset = offset
        self._tokenizer = tokenizer

    @cached_property
    def text(self) -> str:
        token = cast(str, self._tokenizer.convert_ids_to_tokens(self.id))
        return self._tokenizer.convert_tokens_to_string([token])

    @property
    def offset(self) -> int:
        return self._offset
