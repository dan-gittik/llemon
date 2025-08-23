from functools import cached_property
from typing import Sequence

from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from llemon.apis.llm.llm_model import LLMModel
from llemon.apis.llm.llm_tokenizer import LLMTokenizer, LLMToken

TOKENIZERS: dict[str, PreTrainedTokenizerFast] = {}


class HuggingFaceTokenizer(LLMTokenizer):

    def __init__(self, model: LLMModel) -> None:
        self.model = model
        if model.name not in TOKENIZERS:
            TOKENIZERS[model.name] = AutoTokenizer.from_pretrained(model.name, use_fast=False)
        self._tokenizer = TOKENIZERS[model.name]
    
    async def count(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False, return_offset_mapping=False))
    
    async def parse(self, text: str) -> Sequence[LLMToken]:
        output = self._tokenizer(text, add_special_tokens=False, return_offset_mapping=True)
        tokens: list[HuggingFaceToken] = []
        for id, (start, end) in zip(output["input_ids"], output["offset_mapping"]):
            token = HuggingFaceToken(id, start, self._tokenizer)
            tokens.append(token)
        return tokens
    
    async def encode(self, *texts: str) -> list[int]:
        output = self._tokenizer(texts, add_special_tokens=False, return_token_type_ids=False)
        return output["input_ids"]
    
    async def decode(self, ids: list[int]) -> str:
        return "".join(self._tokenizer.convert_ids_to_pieces(ids))


class HuggingFaceToken(LLMToken):

    def __init__(self, id: int, offset: int, tokenizer: PreTrainedTokenizerFast) -> None:
        self.id = id
        self._offset = offset
        self._tokenizer = tokenizer
    
    @cached_property
    def text(self) -> str:
        return self._tokenizer.convert_ids_to_pieces(self.id)[0]
    
    @property
    def offset(self) -> int:
        return self._offset