import os
from functools import cached_property
from typing import ClassVar, Sequence, cast

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from llemon.sync.llm_model import LLMModel
from llemon.sync.count_tokens import count_tokens
from llemon.sync.llm_tokenizer import LLMToken, LLMTokenizer
from llemon.sync.generate import GenerateRequest

TOKENIZERS: dict[str, PreTrainedTokenizerFast] = {}


class HFTokenizer(LLMTokenizer):

    label: ClassVar[str] = "hf"

    def __init__(self, model: LLMModel) -> None:
        super().__init__(model)
        if model.name not in TOKENIZERS:
            TOKENIZERS[model.name] = AutoTokenizer.from_pretrained(model.name, use_fast=False)
        self._tokenizer = TOKENIZERS[model.name]

    def parse(self, text: str) -> Sequence[LLMToken]:
        output = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens: list[HuggingFaceToken] = []
        for id, (start, end) in zip(output["input_ids"], output["offset_mapping"]):
            token = HuggingFaceToken(id, start, self._tokenizer)
            tokens.append(token)
        return tokens

    def encode(self, *texts: str) -> list[int]:
        output = self._tokenizer(texts, add_special_tokens=False)
        return output["input_ids"][0]

    def decode(self, *ids: int) -> str:
        tokens = cast(list[str], self._tokenizer.convert_ids_to_tokens(list(ids)))
        return self._tokenizer.convert_tokens_to_string(tokens)

    def _count(self, request: GenerateRequest) -> int:
        return count_tokens(request, self.__count)

    def __count(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])


class HuggingFaceToken(LLMToken):

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
